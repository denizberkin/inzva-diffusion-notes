from manim import (
    Scene, Text, MathTex, Axes, VGroup, Line, Dot, 
    SurroundingRectangle, ValueTracker, always_redraw,
    FadeIn, FadeOut, Create, Write, linear, YELLOW, BOLD,
    DOWN, UP, LEFT, RIGHT, DashedLine
)
import numpy as np


class KLDivergenceScene(Scene):
    def construct(self):
        fg = "#EAEAEA"          # primary text
        axis_col = "#888888"    # subdued axes

        teal = "#2EE6C6"        # brighter for dark bg
        magenta = "#FF4FA3"

        light_teal = "#2EE6C6"
        light_magenta = "#FF4FA3"

        highlight = "#FFC857"   # amber (better than yellow)

        # ------------------------------------------------------------
        # Title
        # ------------------------------------------------------------
        title = Text("KL-Divergence", font_size=42, color=fg, weight=BOLD)
        title.to_edge(UP, buff=0.35)

        subtitle = Text(
            "How much distribution Q differs from distribution P",
            font_size=24,
            color=fg,
        )
        subtitle.next_to(title, DOWN, buff=0.15)

        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(0.5)

        # ============================================================
        # 1. DISCRETE KL
        # ============================================================
        discrete_formula = MathTex(
            r"\mathrm{KL}(P \Vert Q)",
            r"=",
            r"\sum_{i=1}^{n}",
            r"p_i",
            r"\log\left(",
            r"\frac{p_i}{q_i}",
            r"\right)",
            font_size=46,
            color=fg,
        )
        discrete_formula.set_color_by_tex("p_i", teal)
        discrete_formula.set_color_by_tex("q_i", magenta)
        discrete_formula.next_to(subtitle, DOWN, buff=0.45)

        self.play(Write(discrete_formula))
        self.wait(0.5)

        discrete_group = self.make_discrete_plot(
            teal, magenta, axis_col, fg
        )
        discrete_group.next_to(discrete_formula, DOWN, buff=0.5)

        self.play(Create(discrete_group))
        self.wait(0.5)

        # Highlight ratio term
        ratio_box = SurroundingRectangle(
            discrete_formula[5],
            color=highlight,
            buff=0.08)
        explanation = Text(
            "Each bin contributes according to the mismatch ratio",
            font_size=24,
        )
        explanation.next_to(discrete_group, DOWN, buff=0.35)

        self.play(Create(ratio_box), FadeIn(explanation))
        self.wait(0.8)

        # Animate few selected discrete bars
        bars_p, bars_q = discrete_group.bars_p, discrete_group.bars_q

        for idx in [2, 7, 13, 20, 27]:
            p_bar = bars_p[idx]
            q_bar = bars_q[idx]

            p_box = SurroundingRectangle(p_bar, color=teal, buff=0.04)
            q_box = SurroundingRectangle(q_bar, color=magenta, buff=0.04)

            local_term = MathTex(
                rf"p_{{{idx+1}}}\log\left(\frac{{p_{{{idx+1}}}}}{{q_{{{idx+1}}}}}\right)",
                font_size=30,
            )
            local_term.next_to(explanation, DOWN, buff=0.2)

            self.play(Create(p_box), Create(q_box), FadeIn(local_term), run_time=0.4)
            self.wait(0.25)
            self.play(FadeOut(p_box), FadeOut(q_box), FadeOut(local_term), run_time=0.3)

        self.wait(0.5)

        # Clear discrete
        self.play(
            FadeOut(discrete_formula),
            FadeOut(discrete_group),
            FadeOut(ratio_box),
            FadeOut(explanation),
        )

        # ============================================================
        # 2. CONTINUOUS KL
        # ============================================================
        continuous_formula = MathTex(
            r"\mathrm{KL}(P \Vert Q)",
            r"=",
            r"\int",
            r"p(x)",
            r"\log\left(",
            r"\frac{p(x)}{q(x)}",
            r"\right)",
            r"\,dx",
            r"=",
            r"\mathbb{E}_{x\sim p}",
            r"\left[",
            r"\log\left(\frac{p(x)}{q(x)}\right)",
            r"\right]",
            font_size=42,
            color=fg,
        )
        continuous_formula.set_color_by_tex("p(x)", teal)
        continuous_formula.set_color_by_tex("q(x)", magenta)
        continuous_formula.next_to(subtitle, DOWN, buff=0.45)

        self.play(Write(continuous_formula))
        self.wait(0.5)

        continuous_group = self.make_continuous_plot(
            teal, magenta, light_teal, light_magenta, axis_col, fg
        )
        continuous_group.next_to(continuous_formula, DOWN, buff=0.45)

        p_curve = continuous_group.p_curve
        q_curve = continuous_group.q_curve
        p_area = continuous_group.p_area
        q_area = continuous_group.q_area

        self.play(Create(continuous_group.axes_group))
        self.play(FadeIn(p_area), Create(p_curve))
        self.play(FadeIn(q_area), Create(q_curve))
        self.wait(0.5)

        expectation_box = SurroundingRectangle(
            continuous_formula[9:],
            color=YELLOW,
            buff=0.08,
        )

        expectation_text = Text(
            "The integral is an expectation over samples from P",
            font_size=24,
        )
        expectation_text.next_to(continuous_group, DOWN, buff=0.3)

        self.play(Create(expectation_box), FadeIn(expectation_text))
        self.wait(0.5)

        # Moving sample x
        tracker = ValueTracker(0.5)

        def get_x():
            return tracker.get_value()

        def p_func(x):
            return continuous_group.p_func(x)

        def q_func(x):
            return continuous_group.q_func(x)

        p_axes = continuous_group.p_axes
        q_axes = continuous_group.q_axes

        vertical_p = always_redraw(
            lambda: DashedLine(
                p_axes.c2p(get_x(), 0),
                p_axes.c2p(get_x(), p_func(get_x())),
                color=teal,
                stroke_width=4,
            )
        )

        vertical_q = always_redraw(
            lambda: DashedLine(
                q_axes.c2p(get_x(), 0),
                q_axes.c2p(get_x(), q_func(get_x())),
                color=magenta,
                stroke_width=4,
            )
        )

        dot_p = always_redraw(
            lambda: Dot(
                p_axes.c2p(get_x(), p_func(get_x())),
                color=teal,
                radius=0.055,
            )
        )

        dot_q = always_redraw(
            lambda: Dot(
                q_axes.c2p(get_x(), q_func(get_x())),
                color=magenta,
                radius=0.055,
            )
        )

        local_ratio = always_redraw(
            lambda: MathTex(
                r"\log\left(\frac{p(x)}{q(x)}\right)",
                font_size=30,
            ).next_to(continuous_group, DOWN, buff=0.9)
        )

        self.play(
            FadeIn(vertical_p),
            FadeIn(vertical_q),
            FadeIn(dot_p),
            FadeIn(dot_q),
            FadeIn(local_ratio),
        )

        self.play(tracker.animate.set_value(9.3), run_time=5.0, rate_func=linear)
        self.wait(0.8)

        final_note = Text(
            "KL is zero only when P and Q match everywhere.",
            font_size=26,
        )
        final_note.next_to(local_ratio, DOWN, buff=0.25)

        self.play(FadeIn(final_note))
        self.wait(2)

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def make_discrete_plot(self, teal, magenta, axis_col, fg):
        rng = np.random.default_rng(7)
        n = 30

        p_vals = 0.35 + 0.55 * rng.random(n)
        q_vals = 0.20 + 0.75 * rng.random(n)

        # Smooth-ish structure
        p_vals = 0.55 + 0.25 * np.sin(np.linspace(0, 2.8 * np.pi, n)) + 0.15 * rng.random(n)
        q_vals = 0.50 + 0.30 * np.sin(np.linspace(-1.5, 2.6 * np.pi - 1.5, n)) + 0.12 * rng.random(n)

        p_vals = np.clip(p_vals, 0.15, 0.95)
        q_vals = np.clip(q_vals, 0.10, 0.95)

        width = 9.5
        height = 1.35
        
        axes_cfg = {"color": axis_col, "stroke_width": 2}

        p_axes = Axes(
            x_range=[0, n + 1, 5],
            y_range=[0, 1.05, 0.5],
            x_length=width,
            y_length=height,
            tips=True,
            axis_config=axes_cfg,
        )

        q_axes = p_axes.copy().next_to(p_axes, DOWN, buff=0.25)

        p_label = MathTex("p_i", font_size=34, color=teal)
        p_label.next_to(p_axes.y_axis.get_top(), LEFT, buff=0.15)

        q_label = MathTex("q_i", font_size=34, color=magenta)
        q_label.next_to(q_axes.y_axis.get_top(), LEFT, buff=0.15)

        i_label_1 = MathTex("i", font_size=34)
        i_label_1.next_to(p_axes.x_axis.get_end(), RIGHT, buff=0.15)

        i_label_2 = MathTex("i", font_size=34)
        i_label_2.next_to(q_axes.x_axis.get_end(), RIGHT, buff=0.15)

        bars_p = VGroup()
        bars_q = VGroup()

        for i in range(n):
            x = i + 1

            line_p = Line(
                p_axes.c2p(x, 0),
                p_axes.c2p(x, p_vals[i]),
                color=teal,
                stroke_width=6,
            ).set_opacity(0.25)

            dot_p = Dot(
                p_axes.c2p(x, p_vals[i]),
                color=teal,
                radius=0.045,
            )

            bar_p = VGroup(line_p, dot_p)
            bars_p.add(bar_p)

            line_q = Line(
                q_axes.c2p(x, 0),
                q_axes.c2p(x, q_vals[i]),
                color=magenta,
                stroke_width=6,
            ).set_opacity(0.25)

            dot_q = Dot(
                q_axes.c2p(x, q_vals[i]),
                color=magenta,
                radius=0.045,
            )

            bar_q = VGroup(line_q, dot_q)
            bars_q.add(bar_q)

        group = VGroup(
            p_axes,
            q_axes,
            p_label,
            q_label,
            i_label_1,
            i_label_2,
            bars_p,
            bars_q,
        )

        group.bars_p = bars_p
        group.bars_q = bars_q

        return group

    def make_continuous_plot(self, teal, magenta, light_teal, light_magenta, axis_col, fg):
        width = 9.5
        height = 1.5
        
        axes_cfg = {
            "color": axis_col,
            "stroke_width": 2,
        }

        p_axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 1.3, 0.5],
            x_length=width,
            y_length=height,
            tips=True,
            axis_config=axes_cfg,
        )

        q_axes = p_axes.copy().next_to(p_axes, DOWN, buff=0.35)

        def p_func(x):
            return (
                0.55
                + 0.18 * np.sin(0.8 * x)
                + 0.12 * np.sin(2.1 * x + 0.4)
                + 0.42 * np.exp(-((x - 7.8) ** 2) / 0.10)
                + 0.16 * np.exp(-((x - 1.2) ** 2) / 0.25)
            )

        def q_func(x):
            return (
                0.45
                + 0.26 * np.sin(0.65 * x - 0.8)
                + 0.10 * np.sin(2.0 * x - 1.2)
                + 0.34 * np.exp(-((x - 4.4) ** 2) / 0.20)
                + 0.20 * np.exp(-((x - 8.7) ** 2) / 0.18)
            )

        p_curve = p_axes.plot(p_func, x_range=[0, 10], color=teal, stroke_width=4)
        q_curve = q_axes.plot(q_func, x_range=[0, 10], color=magenta, stroke_width=4)

        p_area = p_axes.get_area(
            p_curve,
            x_range=[0, 10],
            color=light_teal,
            opacity=0.75,
        )

        q_area = q_axes.get_area(
            q_curve,
            x_range=[0, 10],
            color=light_magenta,
            opacity=0.75,
        )

        p_label = MathTex("p(x)", font_size=34, color=teal)
        p_label.next_to(p_axes.y_axis.get_top(), LEFT, buff=0.15)

        q_label = MathTex("q(x)", font_size=34, color=magenta)
        q_label.next_to(q_axes.y_axis.get_top(), LEFT, buff=0.15)

        x_label_1 = MathTex("x", font_size=34)
        x_label_1.next_to(p_axes.x_axis.get_end(), RIGHT, buff=0.15)

        x_label_2 = MathTex("x", font_size=34)
        x_label_2.next_to(q_axes.x_axis.get_end(), RIGHT, buff=0.15)

        axes_group = VGroup(
            p_axes,
            q_axes,
            p_label,
            q_label,
            x_label_1,
            x_label_2,
        )

        group = VGroup(
            axes_group,
            p_area,
            q_area,
            p_curve,
            q_curve,
        )

        group.axes_group = axes_group
        group.p_axes = p_axes
        group.q_axes = q_axes
        group.p_curve = p_curve
        group.q_curve = q_curve
        group.p_area = p_area
        group.q_area = q_area
        group.p_func = p_func
        group.q_func = q_func

        return group