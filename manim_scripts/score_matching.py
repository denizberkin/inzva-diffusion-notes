from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from manim import (
    MovingCameraScene,
    Arrow,
    BOLD,
    DOWN,
    FadeIn,
    GrowArrow,
    ImageMobject,
    LaggedStart,
    MathTex,
    RoundedRectangle,
    Text,
    VGroup,
    VMobject,
    Dot,
    Create,
    MoveAlongPath,
    Write,
    config,
)


config.background_color = "#000000"


# Analytic 2D data density: a two-mode Gaussian mixture.
# Score field:
#
#     s(x) = \nabla_x log p_data(x)
#
# For an isotropic Gaussian component, the local score points toward
# the component mean. For the mixture, the score is the posterior-weighted
# average of those component scores.

CENTERS = np.array(
    [
        [1.65, 1.05],
        [1.65, -1.25],
    ],
    dtype=np.float64,
)

SIGMAS = np.array([0.62, 0.58], dtype=np.float64)
WEIGHTS = np.array([0.53, 0.47], dtype=np.float64)


def mixture_density_and_score(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    xy:
        Array with shape (N, 2).

    Returns
    -------
    p:
        Mixture density, shape (N,).
    score:
        $\nabla_x log p(x)$, shape (N, 2).
    """
    xy = np.asarray(xy, dtype=np.float64)

    diff = xy[:, None, :] - CENTERS[None, :, :]
    sq_dist = np.sum(diff**2, axis=-1)

    var = SIGMAS**2
    norm_const = 1.0 / (2.0 * np.pi * var)

    component_density = (
        WEIGHTS[None, :]
        * norm_const[None, :]
        * np.exp(-0.5 * sq_dist / var[None, :])
    )

    p = component_density.sum(axis=1)

    component_scores = (CENTERS[None, :, :] - xy[:, None, :]) / var[None, :, None]

    score = (
        component_density[:, :, None] * component_scores
    ).sum(axis=1) / (p[:, None] + 1e-12)

    return p, score


def colormap(z: np.ndarray) -> np.ndarray:
    """
    Dark-purple -> magenta -> coral -> orange -> pale-yellow.
    """
    anchors = np.linspace(0.0, 1.0, 7)

    colors = np.array(
        [
            [4, 3, 18],
            [27, 15, 69],
            [77, 22, 121],
            [148, 41, 122],
            [226, 67, 93],
            [255, 142, 91],
            [255, 231, 158],
        ],
        dtype=np.float64,
    ) / 255.0

    rgb = np.zeros((*z.shape, 3), dtype=np.float64)

    for c in range(3):
        rgb[..., c] = np.interp(z, anchors, colors[:, c])

    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def make_density_image(
    path: str | Path,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    width: int = 1600,
) -> None:
    """
    Creates a contour-like density background as a PNG.
    """
    path = Path(path)

    aspect = config.frame_height / config.frame_width
    height = int(width * aspect)

    x_min, x_max = x_range
    y_min, y_max = y_range

    xs = np.linspace(x_min, x_max, width)
    ys = np.linspace(y_max, y_min, height)

    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    p, _ = mixture_density_and_score(points)
    logp = np.log(p + 1e-12).reshape(height, width)

    lo = np.percentile(logp, 4)
    hi = np.percentile(logp, 99.7)

    z = (logp - lo) / (hi - lo + 1e-12)
    z = np.clip(z, 0.0, 1.0)

    # Discrete bands give the contour-field look.
    n_bands = 14
    z = np.floor(z * n_bands) / n_bands

    # Slight gamma adjustment.
    z = z**0.92

    rgb = colormap(z)

    Image.fromarray(rgb).save(path)


class ScoreMatchingVectorField(MovingCameraScene):
    def construct(self):
        # Match coordinate system to the Manim frame aspect ratio.
        x_min, x_max = -4.0, 4.0
        y_half = (x_max - x_min) * config.frame_height / config.frame_width / 2.0
        y_min, y_max = -y_half, y_half

        density_path = Path("score_matching_density_field.png")

        make_density_image(
            density_path,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            width=1800,
        )

        density_img = ImageMobject(str(density_path))
        density_img.set(width=config.frame_width)
        density_img.set_opacity(0.98)
        density_img.move_to([0, 0, 0])

        def to_scene(point: np.ndarray | tuple[float, float]) -> np.ndarray:
            x, y = point
            sx = ((x - x_min) / (x_max - x_min) - 0.5) * config.frame_width
            sy = ((y - y_min) / (y_max - y_min) - 0.5) * config.frame_height
            return np.array([sx, sy, 0.0])

        # Score vector field.
        rng = np.random.default_rng(7)

        arrows = VGroup()

        xs = np.linspace(x_min + 0.35, x_max - 0.35, 13)
        ys = np.linspace(y_min + 0.30, y_max - 0.30, 8)

        p_max, _ = mixture_density_and_score(CENTERS)

        for x in xs:
            for y in ys:
                # avoid perfect grid
                jitter = rng.uniform(-0.045, 0.045, size=2)
                pos = np.array([x, y]) + jitter

                p, score = mixture_density_and_score(pos[None, :])
                score = score[0]

                norm = np.linalg.norm(score)
                if norm < 1e-8:
                    continue

                direction = score / norm

                density_strength = np.clip(np.sqrt(p[0] / p_max.max()), 0.0, 1.0)

                length = 0.18 + 0.50 * density_strength + 0.08 * np.tanh(norm)
                stroke_width = 2.2 + 2.3 * density_strength
                opacity = 0.42 + 0.58 * density_strength

                scene_pos = to_scene(pos)
                start = scene_pos - np.array([direction[0], direction[1], 0.0]) * length * 0.5
                end = scene_pos + np.array([direction[0], direction[1], 0.0]) * length * 0.5

                arrow = Arrow(
                    start=start,
                    end=end,
                    buff=0.0,
                    color="#ffffff",
                    stroke_width=stroke_width,
                    max_tip_length_to_length_ratio=0.30,
                    max_stroke_width_to_length_ratio=8.0,
                )
                arrow.set_opacity(opacity)
                arrows.add(arrow)

        # Labels.
        title = Text("Score Matching", font_size=34, weight=BOLD, color="#f3f0ff")

        formula = MathTex(
            r"s_\theta(x) \approx \nabla_x \log p_{\mathrm{data}}(x)",
            font_size=34,
            color="#f7f3ff",
        )

        subtitle = Text(
            "vectors point toward higher probability density",
            font_size=19,
            color="#d9d0ff",
        )

        label_group = VGroup(title, formula, subtitle)
        label_group.arrange(DOWN, aligned_edge="LEFT", buff=0.16)
        label_group.to_corner(np.array([-1, 1, 0]), buff=0.38)

        panel = RoundedRectangle(
            corner_radius=0.18,
            width=label_group.width + 0.55,
            height=label_group.height + 0.42,
        )
        panel.move_to(label_group)
        panel.set_fill("#05030e", opacity=0.62)
        panel.set_stroke("#ffffff", opacity=0.12, width=1.0)

        # Mode markers.
        mode_dots = VGroup(
            *[
                Dot(
                    point=to_scene(center),
                    radius=0.055,
                    color="#fff4b8",
                )
                for center in CENTERS
            ]
        )

        # Optional denoising / score-ascent trajectory.
        # This shows how following the score moves a noisy point toward a mode.
        def score_ascent_path(
            start: np.ndarray,
            steps: int = 18,
            step_size: float = 0.16,
        ) -> list[np.ndarray]:
            current = start.astype(np.float64)
            path = [current.copy()]

            for _ in range(steps):
                _, score = mixture_density_and_score(current[None, :])
                direction = score[0] / (np.linalg.norm(score[0]) + 1e-12)
                current = current + step_size * direction
                path.append(current.copy())

            return path

        trajectory_points = score_ascent_path(np.array([-2.75, -0.35]))

        trajectory = VMobject()
        trajectory.set_points_smoothly([to_scene(p) for p in trajectory_points])
        trajectory.set_stroke("#8ee6ff", width=4.0, opacity=0.95)

        moving_dot = Dot(
            point=to_scene(trajectory_points[0]),
            radius=0.065,
            color="#8ee6ff",
        )

        trajectory_label = Text(
            "follow the score",
            font_size=18,
            color="#bdf3ff",
        )
        trajectory_label.next_to(trajectory, DOWN, buff=0.18)

        # Animation.
        self.play(FadeIn(density_img), run_time=1.0)
        self.play(FadeIn(panel), Write(label_group), run_time=1.0)

        self.play(
            LaggedStart(
                *[GrowArrow(arrow) for arrow in arrows],
                lag_ratio=0.012,
            ),
            run_time=2.3,
        )

        self.play(Create(mode_dots), run_time=0.6)

        self.play(
            Create(trajectory),
            FadeIn(moving_dot),
            FadeIn(trajectory_label),
            run_time=0.8,
        )

        self.play(
            MoveAlongPath(moving_dot, trajectory),
            run_time=2.2,
        )

        self.wait(1.5)