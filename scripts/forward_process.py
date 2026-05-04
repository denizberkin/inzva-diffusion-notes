from manim import (
    Scene, MathTex, ImageMobject,
    FadeIn, ReplacementTransform,
    UP, DOWN, ORIGIN,
)
from PIL import Image
import numpy as np
import os


class ForwardDiffusionNoise(Scene):
    def construct(self):
        # Config
        image_path = "assets/bmo_space.png"
        num_steps = 40
        update_every = 0.08
        display_height = 4.2
        seed = 42

        rng = np.random.default_rng(seed)

        # Load image
        img = Image.open(image_path).convert("RGB")
        img_np = np.asarray(img).astype(np.float32) / 255.0

        # Save temporary noisy frames
        os.makedirs("tmp_diffusion_frames", exist_ok=True)

        # Text objects
        formula = MathTex(
            r"x_t \sim q(x_t \mid x_{t-1})"
        ).scale(0.9).to_edge(UP)

        step_label = MathTex(r"x_0").scale(0.9).to_edge(DOWN)

        # Initial image
        initial_frame_path = "tmp_diffusion_frames/frame_000.png"
        Image.fromarray((img_np * 255).astype(np.uint8)).save(initial_frame_path)

        image_mobject = ImageMobject(initial_frame_path)
        image_mobject.height = display_height
        image_mobject.move_to(ORIGIN)

        self.play(FadeIn(formula), FadeIn(image_mobject), FadeIn(step_label))
        self.wait(0.5)

        # Forward diffusion visualization
        current_image = image_mobject
        current_label = step_label

        for t in range(1, num_steps + 1):
            alpha_bar_t = 1.0 - (t / num_steps)
            alpha_bar_t = max(alpha_bar_t, 0.0)

            noise = rng.normal(loc=0.0, scale=1.0, size=img_np.shape)

            noisy_img = (
                np.sqrt(alpha_bar_t) * img_np
                + np.sqrt(1.0 - alpha_bar_t) * noise
            )

            noisy_img = np.clip(noisy_img, 0.0, 1.0)

            frame_path = f"tmp_diffusion_frames/frame_{t:03d}.png"
            Image.fromarray((noisy_img * 255).astype(np.uint8)).save(frame_path)

            new_image = ImageMobject(frame_path)
            new_image.height = display_height
            new_image.move_to(ORIGIN)

            new_label = MathTex(rf"x_{{{t}}}").scale(0.9).to_edge(DOWN)

            self.play(
                ReplacementTransform(current_image, new_image),
                ReplacementTransform(current_label, new_label),
                run_time=update_every,
            )

            current_image = new_image
            current_label = new_label

        self.wait(1.0)