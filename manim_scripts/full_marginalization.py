from manim import (
    DOWN, LEFT, RIGHT, UP,
    FadeOut, LaggedStart, Create, Write, FadeIn, Transform,
    VGroup, Scene, Axes,
    Circle, Dot, MathTex, Rectangle, SurroundingRectangle, DashedLine,
    WHITE, TEAL_A, TEAL_B, TEAL_C, TEAL_D, TEAL_E, YELLOW,
    linear,
)
import numpy as np


class FullMarginalization(Scene):
    def construct(self):
        left_origin = LEFT * 2.8 + DOWN * 0.9
        plot_size = 4.6

        axes = Axes(
            x_range=[-3, 3.5, 1],
            y_range=[-3, 3.5, 1],
            x_length=plot_size,
            y_length=plot_size,
            tips=True,
            axis_config={"color": WHITE},
        ).move_to(left_origin)

        x_label = MathTex("x_1").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = MathTex("x_2").next_to(axes.y_axis.get_end(), UP).shift(DOWN * 0.1)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # Joint Gaussian contours
        contours = VGroup()
        for r, c, op in zip(
            [0.6, 1.1, 1.6, 2.1, 2.6, 3.0],
            [TEAL_A, TEAL_B, TEAL_C, TEAL_D, TEAL_E],
            [0.22, 0.17, 0.12, 0.08, 0.04],
        ):
            circ = Circle(
                radius=axes.x_axis.unit_size * r,
                color=c,
                stroke_width=3,
                fill_color=c,
                fill_opacity=op,
            ).move_to(axes.c2p(0, 0))
            contours.add(circ)

        joint_label = MathTex(r"p(x_1, x_2)").scale(0.75)
        joint_label.next_to(axes, LEFT, buff=0.15).shift(UP * 0.6 + RIGHT * 0.4)

        self.play(
            LaggedStart(*[Create(c) for c in contours], lag_ratio=0.08),
            Write(joint_label),
        )

        # Marginal graph above joint plot
        marginal_axes = Axes(
            x_range=[-3, 3.5, 1],
            y_range=[0, 0.45, 0.2],
            x_length=plot_size,
            y_length=1.55,
            tips=False,
            axis_config={"color": WHITE},
        )
        marginal_axes.next_to(axes, UP, buff=0.45)

        marginal_label = MathTex(r"p(x_1)=\int p(x_1,x_2)\,dx_2").scale(0.62)
        marginal_label.next_to(x_label, RIGHT)

        self.play(Create(marginal_axes), Write(marginal_label))

        # Gaussian marginal curve p(x1), N(0,1)
        def normal_pdf(x):
            return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

        curve = marginal_axes.plot(
            normal_pdf,
            x_range=[-3, 3],
            color=TEAL_A,
            stroke_width=4,
        )

        # Vertical slice and growing marginal dots
        slice_width = 0.12
        x_values = np.linspace(-2.8, 2.8, 29)

        column = Rectangle(
            width=axes.x_axis.unit_size * slice_width,
            height=axes.y_axis.unit_size * 6,
            color=YELLOW,
            fill_color=YELLOW,
            fill_opacity=0.18,
            stroke_width=2,
        ).move_to(axes.c2p(x_values[0], 0))

        slice_label = MathTex(r"\int p(x_1,x_2)\,dx_2").scale(0.55)
        slice_label.next_to(column, UP, buff=0.15).shift(DOWN).shift(RIGHT)

        self.play(FadeIn(column), Write(slice_label))

        marginal_points = VGroup()
        connector_lines = VGroup()

        for x in x_values:
            px = normal_pdf(x)

            new_col = column.copy().move_to(axes.c2p(x, 0))
            new_label = slice_label.copy().next_to(new_col, UP, buff=0.15).shift(DOWN).shift(RIGHT)

            dot = Dot(
                marginal_axes.c2p(x, px),
                color=TEAL_A,
                radius=0.045,
            )

            connector = DashedLine(
                axes.c2p(x, 3),
                marginal_axes.c2p(x, px),
                color=WHITE,
                stroke_opacity=0.35,
                dash_length=0.06,
            )

            self.play(
                Transform(column, new_col),
                Transform(slice_label, new_label),
                FadeIn(dot),
                FadeIn(connector),
                run_time=0.12,
                rate_func=linear,
            )

            marginal_points.add(dot)
            connector_lines.add(connector)

        self.play(Create(curve), run_time=1.2)
        self.play(FadeOut(column), FadeOut(slice_label), FadeOut(connector_lines))

        final_box = SurroundingRectangle(marginal_label, color=YELLOW, buff=0.12)
        self.play(Create(final_box))
        self.wait(1.5)