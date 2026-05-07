from manim import (
    TEAL_B,
    ThreeDScene,
    BOLD, RIGHT, UP,
    Create, Write, FadeIn,
    ThreeDAxes, Surface, ParametricFunction, Line,
    Text, MathTex,
    WHITE, TEAL_A, BLUE_B,
    TEAL_D, TEAL_E, BLUE_D, BLUE_E,  # noqa: F401
    DEGREES, ManimColor
)

import numpy as np


class JointGaussian3D(ThreeDScene):
    def construct(self):
        title = Text("3D Joint Gaussian and Marginals", font_size=30, weight=BOLD)
        title.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(title))

        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 0.18, 0.05],
            x_length=6,
            y_length=6,
            z_length=2.2,
            axis_config={"color": WHITE},
        )

        x_label = MathTex("x_1").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = MathTex("x_2").next_to(axes.y_axis.get_end(), UP)
        z_label = MathTex(r"p(x_1,x_2)").next_to(axes.z_axis.get_end(), UP)

        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES, zoom=0.8)

        self.play(Create(axes), Write(x_label), Write(y_label), Write(z_label))

        def joint_pdf(u, v):
            return 1 / (2 * np.pi) * np.exp(-0.5 * (u**2 + v**2))


        almost_transparent = ManimColor((1, 1, 1, 0.02))
        surface = Surface(
            lambda u, v: axes.c2p(u, v, joint_pdf(u, v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(36, 36),
            fill_opacity=0.45,
            checkerboard_colors=[TEAL_B, almost_transparent],
            # fill_color=TEAL_B,
            stroke_color=WHITE,
            stroke_width=0.25,
        )

        self.play(Create(surface), run_time=2)

        # Marginal curves
        def normal_pdf(t):
            return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * t**2)

        # Scale marginal curves down visually so they fit with joint z-scale.
        marginal_scale = 0.35

        x_curve = ParametricFunction(
            lambda t: axes.c2p(t, 3.25, marginal_scale * normal_pdf(t)),
            t_range=[-3, 3],
            color=TEAL_A,
            stroke_width=5,
        )

        y_curve = ParametricFunction(
            lambda t: axes.c2p(3.25, t, marginal_scale * normal_pdf(t)),
            t_range=[-3, 3],
            color=BLUE_B,
            stroke_width=5,
        )

        x_base = Line(
            axes.c2p(-3, 3.25, 0),
            axes.c2p(3, 3.25, 0),
            color=TEAL_A,
            stroke_width=2,
        )

        y_base = Line(
            axes.c2p(3.25, -3, 0),
            axes.c2p(3.25, 3, 0),
            color=BLUE_B,
            stroke_width=2,
        )

        px_label = MathTex(r"p(x_1)").scale(0.7)
        px_label.move_to(axes.c2p(1.9, 3.35, 0.17))

        py_label = MathTex(r"p(x_2)").scale(0.7)
        py_label.move_to(axes.c2p(3.4, 1.9, 0.17))

        self.play(
            Create(x_base),
            Create(y_base),
            Create(x_curve),
            Create(y_curve),
            Write(px_label),
            Write(py_label),
            run_time=2,
        )

        # Optional vertical projection slices
        # x_slice = Surface(
        #     lambda u, z: axes.c2p(u, 0, z),
        #     u_range=[-3, 3],
        #     v_range=[0, 0.16],
        #     resolution=(24, 6),
        #     fill_opacity=0.18,
        #     checkerboard_colors=[TEAL_E, TEAL_D],
        #     stroke_width=0.2,
        # )

        # y_slice = Surface(
        #     lambda v, z: axes.c2p(0, v, z),
        #     u_range=[-3, 3],
        #     v_range=[0, 0.16],
        #     resolution=(24, 6),
        #     fill_opacity=0.18,
        #     checkerboard_colors=[BLUE_E, BLUE_D],
        #     stroke_width=0.2,
        # )

        # self.play(FadeIn(x_slice), FadeIn(y_slice), run_time=1)

        self.begin_ambient_camera_rotation(rate=0.22)
        self.wait(8)
        self.stop_ambient_camera_rotation()

        self.wait(1)