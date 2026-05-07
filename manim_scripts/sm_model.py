from manim import (
    LEFT, TEAL_B, Scene, Dot, Line, Arrow, VGroup, Text, MathTex, FadeIn, GrowArrow, Create, LaggedStart,
    GRAY_A, GRAY_B, BLUE_B, GREEN_B, RED_B, ORIGIN, UP, DOWN, MEDIUM, WHITE, MAROON     
) 


class ScoreNetModelDiagram(Scene):
    def construct(self):
        # Styling
        neuron_radius = 0.075
        edge_color = MAROON
        neuron_color = TEAL_B
        input_color = BLUE_B
        sigma_color = GREEN_B
        output_color = RED_B

        label_fs = 22
        tiny_fs = 16

        # Helpers
        def neuron_column(n, x, height=2.8, color=neuron_color):
            ys = [0] if n == 1 else [
                height / 2 - i * height / (n - 1)
                for i in range(n)
            ]

            dots = VGroup(*[
                Dot(
                    point=[x, y, 0],
                    radius=neuron_radius,
                    color=color,
                )
                for y in ys
            ])
            return dots

        def connect_layers(left_layer, right_layer, opacity=0.28):
            edges = VGroup()
            for a in left_layer:
                for b in right_layer:
                    line = Line(
                        a.get_center(),
                        b.get_center(),
                        stroke_width=1.1,
                        color=edge_color,
                    )
                    line.set_opacity(opacity)
                    edges.add(line)
            return edges

        # Network layout
        x_positions = [-5.2, -3.4, -1.6, 0.2, 2.0, 3.8]

        input_layer = neuron_column(3, x_positions[0], height=1.4, color=input_color)
        h1 = neuron_column(7, x_positions[1], height=3.2)
        h2 = neuron_column(7, x_positions[2], height=3.2)
        h3 = neuron_column(7, x_positions[3], height=3.2)
        output_layer = neuron_column(2, x_positions[4], height=0.9, color=output_color)

        # Extra sigma preprocessing node
        sigma_node = Dot(
            point=[-6.3, -1.65, 0],
            radius=neuron_radius,
            color=sigma_color,
        )
        log_sigma_node = Dot(
            point=[-5.2, -1.65, 0],
            radius=neuron_radius,
            color=sigma_color,
        )

        # Edges
        sigma_edge = Arrow(
            sigma_node.get_center(),
            log_sigma_node.get_center(),
            stroke_width=1.5,
            color=sigma_color,
            buff=0.12,
            max_tip_length_to_length_ratio=0.08,
        )

        concat_edge = Arrow(
            log_sigma_node.get_center(),
            input_layer[-1].get_center(),
            stroke_width=1.2,
            color=sigma_color,
            buff=0.12,
            max_tip_length_to_length_ratio=0.08,
        )

        edges = VGroup(
            connect_layers(input_layer, h1),
            connect_layers(h1, h2),
            connect_layers(h2, h3),
            connect_layers(h3, output_layer),
        )

        # Labels
        title = Text(
            "Score Matching Sample Model Architecture",
            font_size=30,
            weight=MEDIUM,
            color=WHITE,
        ).to_edge(UP, buff=0.45)

        input_label = MathTex(
            r"[x_{\mathrm{noisy}}, \log\sigma]",
            font_size=label_fs,
            color=GRAY_A,
        ).next_to(input_layer, DOWN, buff=0.3).shift(LEFT * 0.7)

        sigma_label = MathTex(
            r"\sigma",
            font_size=label_fs,
            color=sigma_color,
        ).next_to(sigma_node, DOWN, buff=0.15)

        log_sigma_label = MathTex(
            r"\log\sigma",
            font_size=label_fs,
            color=sigma_color,
        ).next_to(log_sigma_node, DOWN, buff=0.15)

        h1_label = Text("128", font_size=tiny_fs, color=GRAY_A).next_to(h1, DOWN, buff=0.3)
        h2_label = Text("128", font_size=tiny_fs, color=GRAY_A).next_to(h2, DOWN, buff=0.3)
        h3_label = Text("128", font_size=tiny_fs, color=GRAY_A).next_to(h3, DOWN, buff=0.3)

        output_label = MathTex(
            r"s_\theta(x,\sigma)",
            font_size=label_fs,
            color=GRAY_A,
        ).next_to(output_layer, DOWN, buff=0.3)

        dim_labels = VGroup(
            Text("3", font_size=tiny_fs, color=GRAY_A).next_to(input_layer, UP, buff=0.22),
            Text("2", font_size=tiny_fs, color=GRAY_A).next_to(output_layer, UP, buff=0.22),
        )

        activation_label = Text(
            "SiLU activations between hidden layers",
            font_size=18,
            color=GRAY_B,
        ).to_edge(DOWN, buff=0.45)

        # Group everything
        network = VGroup(
            input_layer,
            h1,
            h2,
            h3,
            output_layer,
            sigma_node,
            log_sigma_node,
            sigma_edge,
            concat_edge,
            edges,
            input_label,
            sigma_label,
            log_sigma_label,
            h1_label,
            h2_label,
            h3_label,
            output_label,
            dim_labels,
        )

        network.move_to(ORIGIN + DOWN * 0.05)

        # Animation
        self.play(FadeIn(title), run_time=0.6)

        self.play(
            FadeIn(input_layer),
            FadeIn(sigma_node),
            FadeIn(log_sigma_node),
            FadeIn(input_label),
            FadeIn(sigma_label),
            FadeIn(log_sigma_label),
            FadeIn(dim_labels[0]),
            run_time=0.8,
        )

        self.play(
            GrowArrow(sigma_edge),
            GrowArrow(concat_edge),
            run_time=0.7,
        )

        self.play(
            LaggedStart(
                FadeIn(h1),
                FadeIn(h2),
                FadeIn(h3),
                FadeIn(output_layer),
                lag_ratio=0.15,
            ),
            run_time=1.0,
        )

        self.play(
            LaggedStart(
                Create(edges[0]),
                Create(edges[1]),
                Create(edges[2]),
                Create(edges[3]),
                lag_ratio=0.18,
            ),
            run_time=1.6,
        )

        self.play(
            FadeIn(h1_label),
            FadeIn(h2_label),
            FadeIn(h3_label),
            FadeIn(output_label),
            FadeIn(dim_labels[1]),
            FadeIn(activation_label),
            run_time=0.8,
        )

        self.wait(2)