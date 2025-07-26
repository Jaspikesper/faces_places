import numpy as np
from manim import *
import os

class MyTwoDimensionalScene(VectorScene):
    def construct(self):
        self.camera.frame_width = 15
        self.camera.frame_height = 15
        w = ValueTracker(2)
        h = 2
        rec = Rectangle(width=w.get_value(), height=h, color=WHITE, fill_color=BLACK, fill_opacity=0.7)
        rec.set_stroke(width=0)
        # Number plane
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
        V1_label = MathTex(r"\begin{bmatrix} 2 \\ 1 \end{bmatrix}").move_to(P1.get_center() + [0.5, 0.5, 0])
        self.play(FadeIn(P1), Create(V1_label))
        self.wait(0.5)
        self.play(Create(V1))
        self.wait(0.5)
        self.play(FadeOut(P1))
        self.wait(0.5)
        # V1_label (no updater)

        # Second vector and label
        vector_2 = np.array([1, 3, 0])
        V2 = Vector(vector_2, color=ORANGE)
        V2_label = MathTex(r"\begin{bmatrix} 1 \\ 3 \end{bmatrix}")
        V2_label.move_to([1.5, 3.5, 0])
        self.play(Create(V2), Create(V2_label))
        self.wait(0.5)


        # Plus sign between V1_label and V2_label
        plus_sign = MathTex(r"\mathbf{+}")
        plus_sign.next_to(V1_label, RIGHT, buff=0.08)
        self.play(V2_label.animate.next_to(plus_sign, RIGHT, buff=0.08)) # Make sure v2 has a place for it to be created
        self.play(Create(plus_sign))
        rec.move_to(plus_sign.get_center())
        self.wait(1)

        # Group equation elements including V1_label
        eqn_group = VGroup(rec, V1_label, plus_sign, V2_label)
        self.play(eqn_group.animate.shift(2 * RIGHT))
        self.wait(0.5)

        self.play(V2.animate.shift(V1.get_end()))
        self.wait(1)

        v3 = vector + vector_2
        V3 = Vector(v3, color=PURPLE)
        equals_sign = MathTex(r"\mathbf{=}").next_to(V2_label, RIGHT, buff=0.12)
        V3_label = MathTex(r"\begin{bmatrix} 3 \\ 4 \end{bmatrix}").next_to(equals_sign, RIGHT, buff=0.12)
        # Dynamic V3_label
        scale_tracker = ValueTracker(1)
        def get_v3_label():
            scaled = scale_tracker.get_value() * vector[:2] + vector_2[:2]
            return MathTex(
                r"\begin{bmatrix} %.1f \\ %.1f \end{bmatrix}" % (scaled[0], scaled[1])
            ).next_to(equals_sign, RIGHT, buff=0.12)
        V3_label_float = get_v3_label().add_updater(lambda m: m.move_to(V3_label.get_center()))
        V3_label_float.add_updater(lambda m: m.become(get_v3_label()))

        rec.add_updater(lambda m: m.stretch_to_fit_width(w.get_value()))
        self.play(Create(equals_sign), Create(V3), Create(V3_label), rec.animate.next_to(equals_sign, LEFT, buff=-0.5), w.animate.set_value(5))
        self.wait(1)
        group_2 = VGroup(rec, V1_label, plus_sign, V2_label, equals_sign, V3_label)
        self.play(group_2.animate.move_to([5.5, 5.5, 0]))
        self.play(group_2.animate.move_to([0, 5.5, 0]))
        self.wait(2.5)

        # Smooth scaling using ValueTracker
        V1.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, scale_tracker.get_value() * vector))
        V2.add_updater(lambda m: m.put_start_and_end_on(scale_tracker.get_value() * vector,
                                                        scale_tracker.get_value() * vector + vector_2))
        V3.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, scale_tracker.get_value() * vector + vector_2))

        # Scalar label
        scalar_label = MathTex(r"\mathbf{1.0} \cdot").next_to(V1_label, LEFT, buff=0.15)
        scalar_label.add_updater(
            lambda m: m.become(
                MathTex(rf"\mathbf{{{scale_tracker.get_value():.1f}}} \cdot").next_to(V1_label, LEFT, buff=0.15)
            )
        )
        V3_label_float.move_to(V3_label.get_center())
        self.play(Create(scalar_label), TransformMatchingTex(V3_label, V3_label_float))

        self.play(scale_tracker.animate.set_value(2), run_time=2.5)
        self.wait(1.5)

        # Remove updaters after animation
        V1.clear_updaters()
        V2.clear_updaters()
        V3.clear_updaters()
        scalar_label.clear_updaters()
        V3_label.clear_updaters()
        scaled_2 = MathTex(r"\mathbf{2}").move_to(scalar_label.get_center())
        scaled_v3_label = MathTex(r"\begin{bmatrix} 5 \\ 5 \end{bmatrix}")
        self.play(TransformMatchingTex(V3_label, scaled_v3_label), TransformMatchingTex(scalar_label, scaled_2))
        self.wait(2.5)
        final_group = VGroup(*{V1, V2, V3, V2_label, plus_sign, equals_sign, rec, scaled_v3_label, scaled_2, V1_label})
        self.play(FadeOut(final_group), run_time=2)
        self.wait(2.5)

if __name__ == '__main__':
    scene = MyTwoDimensionalScene()
    scene.render()
    os.startfile(scene.renderer.file_writer.movie_file_path)