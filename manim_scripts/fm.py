from __future__ import annotations

import numpy as np

from manim import (
    Scene, VGroup, Text, MathTex, Arrow, Dot, Line, RoundedRectangle,
    ValueTracker, always_redraw, FadeIn, Create, Write, 
    BOLD, UP, DOWN, LEFT, RIGHT, smooth,
)


# Config
SEED = 727
N_PARTICLES = 140

BG = "#000000"
PANEL_STROKE = "#2A405C"

TEAL = "#4DD6C1"
TEAL_DARK = "#1B8E84"
YELLOW = "#F2C94C"
ORANGE = "#F2994A"
BLUE = "#5DADEC"
PINK = "#E86BA8"
GRAY = "#AAB7C4"
WHITE_SOFT = "#EAF2F8"


# Helpers
def unit_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


class PanelMapper:
    def __init__(self, center: np.ndarray, scale: float = 0.9):
        self.center = center
        self.scale = scale

    def p2m(self, p: np.ndarray) -> np.ndarray:
        return self.center + self.scale * p[0] * RIGHT + self.scale * p[1] * UP


def make_two_ring_targets(
    rng: np.random.Generator,
    n: int,
    radius: float = 0.75,
    jitter: float = 0.06,
) -> np.ndarray:
    centers = np.array([[-1.35, 0.0], [1.35, 0.0]])
    which = rng.integers(0, 2, size=n)

    theta = rng.uniform(0, 2 * np.pi, size=n)
    r = radius + rng.normal(0.0, jitter, size=n)

    points = centers[which] + np.stack(
        [r * np.cos(theta), r * np.sin(theta)],
        axis=1,
    )
    return points


def mixture_score_field(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    Approximate score of a two-mode Gaussian mixture.

    This is not meant to be a mathematically exact score-matching demo.
    It is a clean visual surrogate:
        score = direction toward regions of high density.
    """
    centers = np.array([[-1.35, 0.0], [1.35, 0.0]])
    var = 0.20 + sigma**2

    diffs = x[None, :] - centers
    energies = -np.sum(diffs**2, axis=1) / (2.0 * var)
    weights = np.exp(energies - np.max(energies))
    weights = weights / (weights.sum() + 1e-8)

    component_scores = -diffs / var
    score = np.sum(weights[:, None] * component_scores, axis=0)
    return score


def flow_velocity_field(x: np.ndarray, t: float) -> np.ndarray:
    """
    Stylized probability-flow velocity field.
    It roughly transports points from a central noise blob into two data modes.
    """
    target_center = np.array([-1.35, 0.0]) if x[0] < 0 else np.array([1.35, 0.0])

    to_mode = target_center - x

    # Small rotational component to make the field look more like transport
    # rather than pure nearest-centroid attraction.
    tangent = np.array([-to_mode[1], to_mode[0]])
    swirl_strength = 0.22 * np.sin(np.pi * t)

    # Early: split mass outward.
    # Late: contract around data modes.
    return to_mode + swirl_strength * tangent


def build_vector_field(
    mapper: PanelMapper,
    field_fn,
    tracker: ValueTracker,
    color: str,
    x_range=(-2.6, 2.6),
    y_range=(-1.8, 1.8),
    step: float = 0.75,
    arrow_length: float = 0.34,
    stroke_width: float = 2.2,
) -> VGroup:
    arrows = VGroup()

    xs = np.arange(x_range[0], x_range[1] + 1e-6, step)
    ys = np.arange(y_range[0], y_range[1] + 1e-6, step)

    value = tracker.get_value()

    for x in xs:
        for y in ys:
            p = np.array([x, y])
            v = field_fn(p, value)
            v = unit_vector(v)

            start = mapper.p2m(p)
            end = mapper.p2m(p + arrow_length * v)

            arr = Arrow(
                start=start,
                end=end,
                buff=0.0,
                max_tip_length_to_length_ratio=0.23,
                stroke_width=stroke_width,
                color=color,
            )
            arr.set_opacity(0.72)
            arrows.add(arr)

    return arrows


def build_dots(
    mapper: PanelMapper,
    points: np.ndarray,
    color: str,
    radius: float = 0.028,
    opacity: float = 0.9,
) -> VGroup:
    dots = VGroup()
    for p in points:
        dot = Dot(
            mapper.p2m(p),
            radius=radius,
            color=color,
        )
        dot.set_opacity(opacity)
        dots.add(dot)
    return dots


# Main Scene
class FlowMatchingVsScoreMatching(Scene):
    def construct(self):
        self.camera.background_color = BG

        rng = np.random.default_rng(SEED)

        # Shared synthetic data distribution.
        x_data = make_two_ring_targets(rng, N_PARTICLES)

        # Score matching noisy samples.
        score_noise = rng.normal(0.0, 1.0, size=x_data.shape)

        # Flow matching source samples.
        x_noise = rng.normal(0.0, 1.0, size=x_data.shape)
        x_noise *= np.array([1.0, 0.72])

        # Trackers.
        sigma_tracker = ValueTracker(1.25)
        flow_t_tracker = ValueTracker(0.0)

        # Panels.
        left_center = LEFT * 3.35 + UP * 0.4
        right_center = RIGHT * 3.35 + UP * 0.4

        left_mapper = PanelMapper(left_center, scale=0.88)
        right_mapper = PanelMapper(right_center, scale=0.88)

        divider = Line(
            start=UP * 3.1,
            end=DOWN * 3.25,
            color=PANEL_STROKE,
            stroke_width=2,
        )
        divider.set_opacity(0.7)

        left_panel = RoundedRectangle(
            width=5.7,
            height=5.55,
            corner_radius=0.22,
            stroke_color=PANEL_STROKE,
            stroke_width=2,
            fill_color="#0C1A2B",
            fill_opacity=0.65,
        ).move_to(left_center)

        right_panel = left_panel.copy().move_to(right_center)

        left_heading = Text(
            "Score Matching",
            font_size=26,
            color=BLUE,
            weight=BOLD,
        ).next_to(left_panel, UP, buff=0.18)

        right_heading = Text(
            "Flow Matching",
            font_size=26,
            color=TEAL,
            weight=BOLD,
        ).next_to(right_panel, UP, buff=0.18)

        left_formula = MathTex(
            r"s_\theta(x,t) \approx \nabla_x \log p_t(x)",
            font_size=24,
            color=WHITE_SOFT,
        ).next_to(left_heading, DOWN, buff=0.1).shift(DOWN * 0.2)

        right_formula = MathTex(
            r"v_\theta(x,t) \approx \frac{d x_t}{dt}",
            font_size=24,
            color=WHITE_SOFT,
        ).next_to(right_heading, DOWN, buff=0.1).shift(DOWN * 0.2)

        self.play(
            FadeIn(left_panel),
            FadeIn(right_panel),
            Create(divider),
            run_time=1.0,
        )
        self.play(
            FadeIn(left_heading),
            FadeIn(right_heading),
            Write(left_formula),
            Write(right_formula),
            run_time=1.1,
        )

        # Score matching side
        def score_points() -> np.ndarray:
            sigma = sigma_tracker.get_value()
            return x_data + sigma * score_noise

        score_dots = always_redraw(
            lambda: build_dots(
                left_mapper,
                score_points(),
                color=BLUE,
                radius=0.022,
                opacity=0.65,
            )
        )

        score_field = always_redraw(
            lambda: build_vector_field(
                left_mapper,
                field_fn=lambda p, s: mixture_score_field(p, sigma=s),
                tracker=sigma_tracker,
                color=YELLOW,
                step=0.72,
                arrow_length=0.32,
                stroke_width=2.0,
            )
        )

        score_label = Text(
            "Noisy data density",
            font_size=19,
            color=GRAY,
        ).move_to(left_mapper.p2m(np.array([0.0, -2.55])))

        score_annotation = Text(
            "learn direction toward higher density",
            font_size=17,
            color=YELLOW,
        ).next_to(score_label, DOWN, buff=0.12)

        self.play(
            FadeIn(score_field),
            FadeIn(score_dots),
            FadeIn(score_label),
            FadeIn(score_annotation),
            run_time=1.2,
        )

        self.play(
            sigma_tracker.animate.set_value(0.28),
            run_time=4.0,
            rate_func=smooth,
        )

        self.wait(0.4)

        # Flow matching side
        def flow_points() -> np.ndarray:
            t = smoothstep(flow_t_tracker.get_value())
            return (1.0 - t) * x_noise + t * x_data

        flow_dots = always_redraw(
            lambda: build_dots(
                right_mapper,
                flow_points(),
                color=TEAL,
                radius=0.022,
                opacity=0.78,
            )
        )

        flow_field = always_redraw(
            lambda: build_vector_field(
                right_mapper,
                field_fn=flow_velocity_field,
                tracker=flow_t_tracker,
                color=ORANGE,
                step=0.72,
                arrow_length=0.32,
                stroke_width=2.0,
            )
        )

        flow_label = Text(
            "Transport from noise to data",
            font_size=19,
            color=GRAY,
        ).move_to(right_mapper.p2m(np.array([0.0, -2.55])))

        flow_annotation = Text(
            "learn velocity along probability paths",
            font_size=17,
            color=ORANGE,
        ).next_to(flow_label, DOWN, buff=0.12)

        self.play(
            FadeIn(flow_field),
            FadeIn(flow_dots),
            FadeIn(flow_label),
            FadeIn(flow_annotation),
            run_time=1.2,
        )

        self.play(
            flow_t_tracker.animate.set_value(1.0),
            run_time=4.2,
            rate_func=smooth,
        )

        self.wait(0.5)

        # Difference summary
        summary_box = RoundedRectangle(
            width=11.4,
            height=1.35,
            corner_radius=0.18,
            stroke_color=PANEL_STROKE,
            stroke_width=1.5,
            fill_color="#0E2236",
            fill_opacity=0.92,
        ).to_edge(DOWN, buff=0.25)

        summary_text = VGroup(
            VGroup(
                Text(
                    "Score matching:",
                    font_size=22,
                    color=BLUE,
                    weight=BOLD,
                ),
                Text(
                    "score field points up density gradients",
                    font_size=22,
                    color=WHITE_SOFT,
                ),
            ).arrange(RIGHT, buff=0.18),

            VGroup(
                Text(
                    "Flow matching:",
                    font_size=22,
                    color=TEAL,
                    weight=BOLD,
                ),
                Text(
                    "velocity field moves mass through time",
                    font_size=22,
                    color=WHITE_SOFT,
                ),
            ).arrange(RIGHT, buff=0.18),
        ).arrange(DOWN, buff=0.16, aligned_edge=LEFT)

        summary_text.move_to(summary_box)

        self.play(
            VGroup(score_label, score_annotation, flow_label, flow_annotation).animate.shift(UP * 0.20),
            VGroup(left_formula, right_formula).animate.shift(DOWN * 0.20),
            left_panel.animate.set(width=5.7, height=5.2),
            right_panel.animate.set(width=5.7, height=5.2),
            FadeIn(summary_box, shift=UP * 0.15),
            FadeIn(summary_text, shift=UP * 0.15),
            run_time=0.9,
        )

        self.wait(2.0)
