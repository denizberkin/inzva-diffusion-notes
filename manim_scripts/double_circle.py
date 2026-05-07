from manim import (
    FadeOut, Scene, Axes, VGroup, Square, Arrow, RoundedRectangle,
    Text, FadeIn, Create, Transform, interpolate_color, smooth,
    ORIGIN, UP, DOWN, LEFT, RIGHT, UL,
    GREY_A, GREY_B, GREY_C, GREY_E, BLACK, WHITE, BOLD,
    ManimColor,
)
import numpy as np


class DiffusionToDoubleRing(Scene):
    def construct(self):
        # Style / config
        self.camera.background_color = "#04080A"
        np.random.seed(7)

        TARGET_COLOR = ManimColor("#72F1E7")
        START_COLOR = ManimColor("#2EC4B6")
        END_COLOR = ManimColor("#FFD166")
        FIELD_COLOR = ManimColor("#5FA8A0")
        BAR_COLOR = ManimColor("#37D6D0")

        # Simulation parameters
        NUM_POINTS = 800
        NUM_STEPS = 26
        DT = 0.20

        # Target geometry: two noisy rings
        R_INNER = 1.75
        R_OUTER = 3.35

        # Axes
        axes = Axes(
            x_range=[-5.2, 5.2, 1],
            y_range=[-3.9, 3.9, 1],
            x_length=11.2,
            y_length=7.6,
            axis_config={
                "color": GREY_B,
                "stroke_opacity": 0.45,
                "stroke_width": 1.4,
                "include_ticks": True,
                "include_tip": False,
            },
        )
        axes.move_to(ORIGIN)

        # Helper functions
        def sample_target_points(n):
            """Sample from a two-ring target distribution."""
            half = n // 2
            angles1 = np.random.uniform(0, 2 * np.pi, half)
            angles2 = np.random.uniform(0, 2 * np.pi, n - half)

            ring1 = np.stack(
                [
                    (R_INNER + np.random.normal(scale=0.08, size=half)) * np.cos(angles1),
                    (R_INNER + np.random.normal(scale=0.08, size=half)) * np.sin(angles1),
                ],
                axis=1,
            )

            ring2 = np.stack(
                [
                    (R_OUTER + np.random.normal(scale=0.10, size=n - half)) * np.cos(angles2),
                    (R_OUTER + np.random.normal(scale=0.10, size=n - half)) * np.sin(angles2),
                ],
                axis=1,
            )

            return np.vstack([ring1, ring2])

        def ring_score(p):
            """
            Final score-like field that pulls a point to the nearest ring.
            """
            x, y = p
            r = np.sqrt(x * x + y * y) + 1e-8

            target_r = R_INNER if abs(r - R_INNER) < abs(r - R_OUTER) else R_OUTER
            radial_dir = np.array([x, y]) / r

            radial_force = 1.35 * (target_r - r) * radial_dir
            center_pull = -0.05 * np.array([x, y])

            return radial_force + center_pull

        def time_dependent_score(p, alpha):
            """
            Blend from a noisy inward+swirl field to the final ring field.
            alpha in [0, 1].
            """
            x, y = p
            r = np.sqrt(x * x + y * y) + 1e-8

            inward = -0.30 * np.array([x, y])

            swirl = 0.95 * (1 - alpha) * np.array([-y, x]) / (r + 1.2)

            wobble = 0.12 * (1 - alpha) * np.array(
                [
                    np.sin(1.6 * y + 4.0 * alpha),
                    np.cos(1.4 * x - 3.0 * alpha),
                ]
            )

            final_ring = ring_score(p)

            return (1 - alpha) * (inward + swirl + wobble) + alpha * final_ring

        def simulate_paths(initial_points, steps, dt):
            """Euler-like integration with decaying noise."""
            states = [initial_points.copy()]
            x = initial_points.copy()

            for k in range(steps):
                alpha = (k + 1) / steps

                drift = np.array([time_dependent_score(p, alpha) for p in x])

                sigma = 0.30 * (1 - alpha) ** 1.2 + 0.015
                noise = np.random.normal(scale=sigma, size=x.shape)

                x = x + dt * drift + noise
                states.append(x.copy())

            return states

        def point_cloud(points, color, side=0.038, opacity=0.95, z_index=5):
            vg = VGroup()
            for px, py in points:
                sq = Square(side_length=side)
                sq.set_stroke(width=0)
                sq.set_fill(color, opacity=opacity)
                sq.move_to(axes.c2p(px, py))
                sq.set_z_index(z_index)
                vg.add(sq)
            return vg

        def target_cloud(points, color=TARGET_COLOR, side=0.030, opacity=0.24):
            vg = VGroup()
            for px, py in points:
                sq = Square(side_length=side)
                sq.set_stroke(width=0)
                sq.set_fill(color, opacity=opacity)
                sq.move_to(axes.c2p(px, py))
                sq.set_z_index(2)
                vg.add(sq)
            return vg

        def make_field(alpha):
            """
            Time-dependent background vector field.
            Recomputed every step so it visibly moves.
            """
            arrows = VGroup()
            xs = np.arange(-5.0, 5.01, 0.95)
            ys = np.arange(-3.5, 3.51, 0.85)

            for x in xs:
                for y in ys:
                    p = np.array([x, y])
                    v = time_dependent_score(p, alpha)
                    norm = np.linalg.norm(v)

                    if norm < 1e-8:
                        continue

                    # Keep arrows visually stable
                    v_vis = 0.52 * v / norm

                    start = axes.c2p(x, y)
                    end = axes.c2p(x + v_vis[0], y + v_vis[1])

                    arrow_color = interpolate_color(FIELD_COLOR, TARGET_COLOR, 0.35 * alpha)

                    arrow = Arrow(
                        start=start,
                        end=end,
                        buff=0,
                        stroke_width=2.0,
                        max_stroke_width_to_length_ratio=8,
                        max_tip_length_to_length_ratio=0.28,
                        color=arrow_color,
                    )
                    arrow.set_opacity(0.20 + 0.10 * (1 - alpha))
                    arrow.set_z_index(1)
                    arrows.add(arrow)

            return arrows

        def legend_panel():
            panel = RoundedRectangle(
                corner_radius=0.16,
                width=4.15,
                height=1.40,
                stroke_color=GREY_C,
                stroke_opacity=0.28,
                fill_color=BLACK,
                fill_opacity=0.42,
            )
            panel.to_corner(UL, buff=0.35)
            panel.set_z_index(20)

            title = Text(
                "Diffusion-like sampling",
                font_size=16,
                color=WHITE,
                weight=BOLD,
            )
            title.move_to(panel.get_center() + UP * 0.35)
            title.set_z_index(21)

            target_marker = Square(side_length=0.09)
            target_marker.set_stroke(width=0)
            target_marker.set_fill(TARGET_COLOR, opacity=0.95)

            target_text = Text(
                "Target distribution",
                font_size=20,
                color=TARGET_COLOR,
            )

            evolving_marker = Square(side_length=0.09)
            evolving_marker.set_stroke(width=0)
            evolving_marker.set_fill(START_COLOR, opacity=0.95)

            evolving_text = Text(
                "Evolving particles",
                font_size=20,
                color=START_COLOR,
            )

            row1 = VGroup(target_marker, target_text).arrange(RIGHT, buff=0.15)
            row2 = VGroup(evolving_marker, evolving_text).arrange(RIGHT, buff=0.15)
            rows = VGroup(row1, row2).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            rows.move_to(panel.get_center() + DOWN * 0.15 + LEFT * 0.08)
            rows.set_z_index(21)

            return VGroup(panel, title, rows)

        def progress_ui(alpha):
            bar_width = 4.2

            bg = RoundedRectangle(
                corner_radius=0.08,
                width=bar_width,
                height=0.18,
                stroke_width=1,
                stroke_color=GREY_C,
                stroke_opacity=0.28,
                fill_color=GREY_E,
                fill_opacity=0.20,
            )
            bg.to_edge(DOWN, buff=0.45)
            bg.shift(RIGHT * 1.5)
            bg.set_z_index(20)

            fill = RoundedRectangle(
                corner_radius=0.08,
                width=max(0.001, bar_width * alpha),
                height=0.18,
                stroke_width=0,
                fill_color=BAR_COLOR,
                fill_opacity=0.95,
            )
            fill.align_to(bg, LEFT)
            fill.move_to(bg.get_left() + RIGHT * (fill.width / 2))
            fill.set_y(bg.get_y())
            fill.set_z_index(21)

            label = Text(
                f"t = {alpha:.2f}",
                font_size=22,
                color=WHITE,
            )
            label.next_to(bg, RIGHT, buff=0.28)
            label.set_z_index(21)

            return VGroup(bg, fill, label)

        # Data
        init_points = np.random.normal(scale=2.75, size=(NUM_POINTS, 2))
        target_points = sample_target_points(NUM_POINTS)
        states = simulate_paths(init_points, NUM_STEPS, DT)

        # Static / initial background elements
        field = make_field(0.0)
        target = target_cloud(target_points)
        legend = legend_panel()
        ui = progress_ui(0.0)

        current_color = interpolate_color(START_COLOR, END_COLOR, 0.0)
        particles = point_cloud(states[0], color=current_color, side=0.038, opacity=0.92)

        subtitle = Text(
            "Random noise  →  Target distribution",
            font_size=20,
            color=GREY_A,
        )
        subtitle.to_edge(UP, buff=0.32).shift(RIGHT * 2.1)
        subtitle.set_z_index(20)

        # Intro
        self.play(Create(axes), run_time=1.1)
        self.play(FadeIn(field), run_time=1.0)
        self.play(FadeIn(target), FadeIn(legend), FadeIn(subtitle), run_time=0.9)
        self.play(FadeIn(particles), FadeIn(ui), run_time=0.8)
        self.wait(0.25)

        # Main evolution
        for i in range(1, len(states)):
            alpha = i / (len(states) - 1)

            new_color = interpolate_color(START_COLOR, END_COLOR, alpha)
            new_particles = point_cloud(
                states[i],
                color=new_color,
                side=0.038,
                opacity=0.95,
            )

            new_field = make_field(alpha)
            new_ui = progress_ui(alpha)

            self.play(
                Transform(particles, new_particles),
                Transform(field, new_field),   # moving vector field
                Transform(ui, new_ui),
                run_time=0.26 if i < len(states) - 1 else 0.40,
                rate_func=smooth,
            )

        # Final hold
        final_text = Text(
            "Final samples align with the two-ring target distribution",
            font_size=24,
            color=END_COLOR,
        )
        final_text.to_edge(DOWN, buff=0.16)
        final_text.set_z_index(20)

        self.play(FadeOut(ui), FadeIn(final_text), run_time=0.7)
        self.wait(2)