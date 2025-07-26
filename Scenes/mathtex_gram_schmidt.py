from manim import *
import os




class Gram(Scene):
    def construct(self):
        eqn1 = MathTex(
            r"\text{We will obtain }",
            r"\{",
            r"\mathbf{v_{1}}",
            r"\perp",
            r"\mathbf{v_{2}}",
            r"\}",
            r"\text{ from }",
            r"\{",
            r"\mathbf{u_{1}}",
            r",",
            r"\mathbf{u_{2}}",
            r"\}",
            r"\text{ using Gram-Schmidt.}"
        )
        eqn2 = MathTex(
            r"\mathbf{v_{1}}",
            r"=",
            r"\mathbf{u_{1}}"
        )
        eqn3 = MathTex(
            r"\mathbf{v_{2}}",
            r"=",
            r"\mathbf{u_{2}}",
            r"-",
            r"\operatorname{proj}_{\mathbf{v_{1}}}\mathbf{u_{2}}"
        )
        eqn4 = MathTex(
            r"\{",
            r"\mathbf{v_{1}}",
            r"\perp",
            r"\mathbf{v_{2}}",
            r"\}",
            r"\text{ from }",
            r"\{",
            r"\mathbf{u_{1}}",
            r",",
            r"\mathbf{u_{2}}",
            r"\}"
        )
        eqn4.move_to([1.7, 2, 0])

        self.play(Create(eqn1))
        self.play(eqn1.animate.move_to([0, 2, 0]))

        self.play(TransformMatchingTex(eqn1.copy(), eqn2))
        self.play(eqn2.animate.move_to([0, 1, 0]))

        self.play(TransformMatchingTex(eqn1.copy(), eqn3))
        self.play(eqn3.animate.move_to([1.3, 0, 0]))

        self.play(TransformMatchingTex(eqn1, eqn4))
        vg = VGroup(eqn2, eqn3, eqn4)
        self.play(vg.animate.scale(0.5).to_corner(UR), run_time=2.5)
        self.wait(2.5)

        # Create number plane


if __name__ == "__main__":
    scene = Gram()
    scene.render()
    config.quality = 'low_quality'
    output_path = r'C:\Users\jaspe\PycharmProjects\PythonProject8\Scenes\media\videos\1080p60\Gram.mp4'
    os.startfile(output_path)