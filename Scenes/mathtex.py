from manim import *
import os

left_side = r"\operatorname{proj}_{\mathbf{v}} \mathbf{u}"
equals = r"="
right_side = r"\frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{v} \rVert^2} \mathbf{v}"

class Gram(Scene):
    def construct(self):
        title = Title("Projection Formula (Unit Vector Simplification)")
        self.play(Create(title))

        left_tex = MathTex(left_side, font_size=48)
        equals_tex = MathTex(equals, font_size=48)
        right_tex = MathTex(right_side, font_size=48)

        eq_group = VGroup(left_tex, equals_tex, right_tex).arrange(RIGHT, buff=0.8)
        eq_group.next_to(title, DOWN, buff=0.8)
        self.play(Write(eq_group))
        self.wait(1)

        # Animate right side moving over left side (should slide!)

        # Optionally, morph the right_tex into left_tex for a smooth finish
        self.play(ReplacementTransform(right_tex, left_tex), FadeOut(equals_tex))
        self.wait(1)
        self.play(left_tex.animate.move_to([0, 2, 0]))
        self.wait(2)



if __name__ == "__main__":
    scene = Gram()
    scene.render()
    config.quality = 'low_quality'
    output_path = r'C:\Users\jaspe\PycharmProjects\PythonProject8\Scenes\media\videos\1080p60\Gram.mp4'
    os.startfile(output_path)