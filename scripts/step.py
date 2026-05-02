from manim import (
    Scene, Text, MathTex, RoundedRectangle, VGroup, Arrow, FadeIn, GrowArrow, UP, DOWN, LEFT, RIGHT,
    Write,
)


class ForwardDiffusionStep(Scene):
    def construct(self):
        self.camera.background_color = "#0b0f14"

        title = Text("Forward Diffusion Step", font_size=40)
        title.to_edge(UP)

        equation = MathTex(
            r"x_t",
            r"=",
            r"\sqrt{1-\beta_t}",
            r"x_{t-1}",
            r"+",
            r"\sqrt{\beta_t}",
            r"\epsilon_t",
            font_size=44,
        )
        equation.next_to(title, DOWN, buff=0.25)

        equation[0].set_color("#9fffd0")
        equation[2].set_color("#b0bec5")
        equation[3].set_color("#8fd3ff")
        equation[5].set_color("#b0bec5")
        equation[6].set_color("#d9b3ff")

        noisy_img = RoundedRectangle(
            width=2.5, height=1.5, corner_radius=0.15,
            color="#8fd3ff", fill_color="#1f3b57", fill_opacity=0.85
        )
        noisy_label = MathTex(r"x_{t-1}", font_size=38).move_to(noisy_img)
        noisy_group = VGroup(noisy_img, noisy_label).shift(LEFT * 4 + DOWN * 0.1)

        noise_img = RoundedRectangle(
            width=2.5, height=1.5, corner_radius=0.15,
            color="#d9b3ff", fill_color="#3b245c", fill_opacity=0.85
        )
        noise_label = MathTex(r"\epsilon_t \sim \mathcal{N}(0,I)", font_size=30).move_to(noise_img)
        noise_group = VGroup(noise_img, noise_label).shift(LEFT * 4 + DOWN * 2)

        signal_coeff = MathTex(r"\times \sqrt{1-\beta_t}", font_size=34, color="#b0bec5")
        signal_coeff.next_to(noisy_group, RIGHT, buff=0.2)

        noise_coeff = MathTex(r"\times \sqrt{\beta_t}", font_size=34, color="#b0bec5")
        noise_coeff.next_to(noise_group, RIGHT, buff=0.2)

        plus = MathTex("+", font_size=58, color="#ffe082")
        plus.move_to(RIGHT * 1.0 + DOWN * 1.05)

        output_box = RoundedRectangle(
            width=2.5, height=1.5, corner_radius=0.15,
            color="#9fffd0", fill_color="#1b4d3e", fill_opacity=0.85
        )
        output_label = MathTex(r"x_t", font_size=38).move_to(output_box)
        output_group = VGroup(output_box, output_label).shift(RIGHT * 4 + DOWN * 1.05)

        arrow1 = Arrow(noisy_group.get_right(), signal_coeff.get_left(), buff=0.1, color="#8fd3ff")
        arrow2 = Arrow(signal_coeff.get_right(), plus.get_left(), buff=0.1, color="#8fd3ff")

        arrow3 = Arrow(noise_group.get_right(), noise_coeff.get_left(), buff=0.1, color="#d9b3ff")
        arrow4 = Arrow(noise_coeff.get_right(), plus.get_left(), buff=0.1, color="#d9b3ff")

        arrow5 = Arrow(plus.get_right(), output_group.get_left(), buff=0.1, color="#9fffd0")

        stochastic_note = Text(
            "Same input + newly sampled noise → different possible xₜ",
            font_size=24,
            color="#cfd8dc",
        )
        stochastic_note.to_edge(DOWN)

        self.play(Write(title))
        self.play(Write(equation))
        self.wait(0.4)

        self.play(FadeIn(noisy_group), FadeIn(noise_group))
        self.play(GrowArrow(arrow1), Write(signal_coeff), GrowArrow(arrow3), Write(noise_coeff))
        self.play(GrowArrow(arrow2), GrowArrow(arrow4), Write(plus))
        self.play(GrowArrow(arrow5), FadeIn(output_group))
        self.play(Write(stochastic_note))

        self.wait(2)