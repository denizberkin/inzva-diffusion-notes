from manim import (
    FadeOut, Scene, Axes, VGroup, ValueTracker, Write,
    Text, MathTex, Dot, Line, DoubleArrow, GrowArrow,
    BLUE, WHITE, UP, DOWN, RIGHT, YELLOW, RED, GREEN, BOLD,
    Create, FadeIn, always_redraw, linear,
)
import numpy as np


class JensensInequalityLog(Scene):
    def construct(self):
        title = Text("Jensen's Inequality for log(x)", font_size=36, weight=BOLD)
        title.to_edge(UP)

        formula = MathTex(
            r"\log(\mathbb{E}[X])",
            r"\geq",
            r"\mathbb{E}[\log X]"
        ).scale(1.1)
        formula.next_to(title, DOWN, buff=0.35)

        self.play(FadeIn(title), Write(formula))
        self.wait(0.2)

        axes = Axes(
            x_range=[0.1, 5.5, 1],
            y_range=[-2.5, 2.0, 1],
            x_length=8,
            y_length=4.8,
            tips=True,
            axis_config={"color": WHITE},
        ).shift(DOWN * 0.6)
        
        x_label = MathTex("x", font_size=24).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = MathTex(r"\log x", font_size=24).next_to(axes.y_axis.get_end(), UP)

        log_curve = axes.plot(
            lambda x: np.log(x),
            x_range=[0.15, 5.2],
            color=BLUE
        )

        curve_label = MathTex(r"f(x)=\log x", color=BLUE)
        curve_label.next_to(log_curve, UP, buff=0.2).shift(DOWN * 0.2)

        self.play(Create(axes), FadeIn(x_label, y_label), FadeOut(formula))
        self.play(Create(log_curve), FadeIn(curve_label))
        self.wait(0.2)

        # Two sample values
        x1_tracker = ValueTracker(1.0)
        x2_tracker = ValueTracker(4.0)
        
        def get_p1():
            x1 = x1_tracker.get_value()
            return axes.c2p(x1, np.log(x1))
        
        def get_p2():
            x2 = x2_tracker.get_value()
            return axes.c2p(x2, np.log(x2))
        
        dot1 = always_redraw(lambda: Dot(get_p1(), color=YELLOW))
        dot2 = always_redraw(lambda: Dot(get_p2(), color=YELLOW))

        label1 = always_redraw(lambda: MathTex(r"x_1").next_to(dot1, DOWN))
        label2 = always_redraw(lambda: MathTex(r"x_2").next_to(dot2, DOWN))
        
        self.play(FadeIn(dot1, label1), FadeIn(dot2, label2))
        
        # --- SECANT ---
        secant = always_redraw(lambda: Line(get_p1(), get_p2(), color=YELLOW))
        self.play(Create(secant))
        
        # --- MEAN POINTS ---
        def get_mean_x():
            return (x1_tracker.get_value() + x2_tracker.get_value()) / 2
        
        def get_mean_logs():
            return (np.log(x1_tracker.get_value()) + np.log(x2_tracker.get_value())) / 2

        def get_log_mean():
            return np.log(get_mean_x())
        
        
        # --- EXPECTATION POINTS ---
        dot_mean_logs = always_redraw(
            lambda: Dot(
                axes.c2p(get_mean_x(), get_mean_logs()),
                color=RED
            )
        )
        
        dot_log_mean = always_redraw(
            lambda: Dot(
                axes.c2p(get_mean_x(), get_log_mean()),
                color=GREEN
            )
        )
        
        self.play(FadeIn(dot_mean_logs, dot_log_mean))
        
        # -- VERTICAL GAP ---
        inequality_gap = always_redraw(
            lambda: DoubleArrow(
                axes.c2p(get_mean_x(), get_mean_logs()),
                axes.c2p(get_mean_x(), get_log_mean()),
                buff=0.05,
                stroke_width=4,
                color=WHITE
            )
        )

        gap_label = MathTex(
            r"\log(\mathbb{E}[X]) - \mathbb{E}[\log X] \geq 0"
        ).scale(0.75)
        gap_label.next_to(inequality_gap, RIGHT, buff=0.25).shift(DOWN * 0.3).shift(RIGHT * 2)

        self.play(GrowArrow(inequality_gap), FadeIn(gap_label))
        self.wait(0.5)

        # --- SLIDING ANIMATION ---
        self.play(
            x1_tracker.animate.set_value(0.5),
            x2_tracker.animate.set_value(5.0),
            run_time=3,
            rate_func=linear
        )

        self.play(
            x1_tracker.animate.set_value(2.0),
            x2_tracker.animate.set_value(3.0),
            run_time=3,
            rate_func=linear
        )

        self.wait(2)
        
        explanation = VGroup(
            Text("For a concave function,", font_size=26),
            MathTex(r"f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]"),
            MathTex(r"\Rightarrow \log(\mathbb{E}[X]) \geq \mathbb{E}[\log X]")
        ).arrange(DOWN, buff=0.25).shift(RIGHT * 2)

        explanation.to_edge(DOWN)

        self.play(FadeIn(explanation))
        self.wait(2)