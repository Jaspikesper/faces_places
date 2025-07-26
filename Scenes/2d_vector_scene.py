import numpy as np
from manim import *
import os

class MyTwoDimensionalScene(VectorScene):
    def construct(self):
        self.camera.frame_width = 15
        self.camera.frame_height = 15

        # Create number plane
        self.number_plane = NumberPlane(
            x_range=[-12, 12, 1],
            y_range=[-12, 12, 1],
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.6,
            }
        )
        self.play(Create(self.number_plane), run_time=2)

        # First vector and label
        vector = np.array([2, 1, 0])
        P1 = Dot(vector)
        V1 = Vector(vector, color=BLUE)
        P1_label = MathTex(r"\begin{pmatrix} 2 \\ 1 \end{pmatrix}").next_to(P1, 0.6*(UP + RIGHT))
        V1_label = MathTex(r"\begin{bmatrix} 2 \\ 1 \end{bmatrix}").move_to(P1_label.get_center())
        self.play(FadeIn(P1))
        self.play(Create(P1_label))
        self.wait(1)
        self.play(Create(V1))
        self.wait(1)
        self.play(FadeOut(P1))
        self.play(TransformMatchingTex(P1_label, V1_label))
        self.wait(1)

        # Second vector and label
        vector_2 = np.array([1, 3, 0])
        V2 = Vector(vector_2, color=ORANGE)
        V2_label = MathTex(r"\begin{bmatrix} 1 \\ 3 \end{bmatrix}").move_to(V2.get_end() + 0.6*RIGHT + 1.16*DOWN)
        self.play(Create(V2))
        self.play(Create(V2_label))

        # Plus sign
        plus_sign = MathTex(r"\mathbf{+}").next_to(V2_label, RIGHT, buff=0.08)
        self.play(Create(plus_sign))
        self.wait(1)

        # Move labels and plus sign two units to the right
        eqn_group = VGroup(V1_label, V2_label, plus_sign)
        self.play(eqn_group.animate.shift(2 * RIGHT))
        self.wait(0.5)

        # Move yellow vector to tip of blue vector
        self.play(V2.animate.shift(V1.get_end()))
        self.wait(1)

        # Result vector and equals sign
        v3 = vector + vector_2
        V3 = Vector(v3, color=PURPLE)
        equals_sign = MathTex(r"\mathbf{=}").next_to(eqn_group, RIGHT, buff=0.12)
        V3_label = MathTex(r"\begin{bmatrix} 3 \\ 4 \end{bmatrix}").move_to(V3.get_end() + 0.75*RIGHT + 2.2*DOWN).add_updater(lambda m: m.next_to(equals_sign, RIGHT, buff=0.12))
        self.play(Create(V3), Create(V3_label), Create(equals_sign))
        self.wait(1)
        group_2 = VGroup(V1_label, V2_label, plus_sign, equals_sign, V3_label)

        # Optional: move equation group to a fixed position
        self.play(group_2.animate.move_to([5.5, 5.5, 0]))
        self.play(group_2.animate.move_to([0, 5.5, 0]))
        self.wait(2.5)

        # Smooth scaling using ValueTracker
        scale_tracker = ValueTracker(1)
        V1.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, scale_tracker.get_value() * vector))
        V2.add_updater(lambda m: m.put_start_and_end_on(scale_tracker.get_value() * vector, scale_tracker.get_value() * vector + vector_2))
        V3.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, scale_tracker.get_value() * vector + vector_2))

        self.play(scale_tracker.animate.set_value(2), run_time=1.5)
        self.wait(1.5)

        # Remove updaters after animation
        V1.clear_updaters()
        V2.clear_updaters()
        V3.clear_updaters()

if __name__ == '__main__':
    scene = MyTwoDimensionalScene()
    scene.render()
    os.startfile(scene.renderer.file_writer.movie_file_path)