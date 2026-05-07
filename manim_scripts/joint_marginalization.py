from manim import (
    BOLD, DOWN, LEFT, RIGHT, UP,
    FadeOut, LaggedStart, Create, Write, FadeIn,
    VGroup, Scene, Axes,
    Text, Circle, Dot, MathTex, Rectangle, SurroundingRectangle, Arrow,
    linear,
    TEAL_A, TEAL_B, TEAL_C, TEAL_D, TEAL_E, YELLOW, WHITE,
    
)


class JointGaussianMarginalization(Scene):
    def construct(self):
        # Layout
        left_origin = LEFT * 3.2 + DOWN * 0.8
        plot_size = 4.8

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=plot_size,
            y_length=plot_size,
            tips=True,
            axis_config={"color": WHITE},
        ).move_to(left_origin)
        
        main_title = Text("Joint Gaussian Marginalization", font_size=32, weight=BOLD).to_edge(UP, buff=0.5)
        self.play(FadeIn(main_title, shift=UP * 0.3))

        x_label = MathTex("x_1").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = MathTex("x_2").next_to(axes.y_axis.get_end(), UP)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # Joint Gaussian contours
        # p(x1, x2), mean=0, covariance=I
        contours = VGroup()
        radii = [0.55, 1.05, 1.55, 2.1]
        colors = [TEAL_B, TEAL_C, TEAL_D, TEAL_E]

        for r, c in zip(radii, colors):
            circ = Circle(
                radius=axes.x_axis.unit_size * r,
                color=c,
                stroke_width=3,
                fill_color=c,
                fill_opacity=0.08,
            ).move_to(axes.c2p(0, 0))
            contours.add(circ)

        center_dot = Dot(axes.c2p(0, 0), color=TEAL_A)

        # p(x1, x2) label: move slightly right
        title = MathTex(r"p(x_1, x_2)").scale(0.8)
        title.next_to(axes, LEFT, buff=0.15).shift(RIGHT * 0.45).shift(UP * 0.5)

        self.play(
            LaggedStart(*[Create(c) for c in contours], lag_ratio=0.12),
            FadeIn(center_dot),
            Write(title),
        )

        # Small square slice: p(x1, x2)
        x_fixed = 0.8
        y_fixed = 1.2
        dx = 0.45
        dy = 0.45

        square = Rectangle(
            width=axes.x_axis.unit_size * dx,
            height=axes.y_axis.unit_size * dy,
            color=YELLOW,
            fill_color=YELLOW,
            fill_opacity=0.25,
            stroke_width=4,
        ).move_to(axes.c2p(x_fixed, y_fixed))

        square_label = MathTex(r"p(x_1, x_2)").scale(0.55)
        square_label.next_to(square, RIGHT, buff=0.15)
        
        square_group = VGroup(square, square_label)

        self.play(Create(square), Write(square_label))
        self.wait(0.5)

        # Highlight conditional column p(x2 | x1)
        column = Rectangle(
            width=axes.x_axis.unit_size * dx,
            height=axes.y_axis.unit_size * 6,
            color=TEAL_A,
            fill_color=TEAL_A,
            fill_opacity=0.22,
            stroke_width=2,
        ).move_to(axes.c2p(x_fixed, 0))

        conditional_label = MathTex(r"p(x_2 \mid x_1)").scale(0.75)
        conditional_label.next_to(column, UP, buff=0.35).shift(RIGHT * 0.15)

        arrow = Arrow(
            conditional_label.get_bottom(),
            square.get_top(),
            buff=0.1,
            color=WHITE,
        )

        self.play(
            y_label.animate.shift(LEFT * 0.5),
            FadeIn(column),
            square.animate.set_fill(YELLOW, opacity=0.45),
            Write(conditional_label),
            Create(arrow),
        )
        self.wait(1.5)

        # Formula on right
        formula = MathTex(
            r"p(x_1)",
            r"=",
            r"\underbrace{\int p(x_1, x_2)\,dx_2}_{\text{sum across all } x_2}",
        ).scale(0.75)

        formula.to_edge(RIGHT, buff=0.55).shift(UP * 0.8)

        # Slide square from top to bottom
        # showing integration over x2
        top_y = 2.6
        bottom_y = -2.6

        moving_square_group = square_group
        
        def move_square_group_to_y(group, square, y):
            target = axes.c2p(x_fixed, y)
            delta = target - square.get_center()
            return group.animate.shift(delta)

        self.play(
            FadeOut(conditional_label, arrow),
            move_square_group_to_y(moving_square_group, square, top_y),
            run_time=0.8,
        )

        self.play(
            Write(formula),
            move_square_group_to_y(moving_square_group, square, bottom_y),
            run_time=3.0,
            rate_func=linear,
        )

        self.wait(0.7)

        # Emphasize formula
        marginal_box = SurroundingRectangle(formula, color=YELLOW, buff=0.15)
        self.play(Create(marginal_box))
        self.wait(1)