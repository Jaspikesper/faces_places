import numpy as np
from pyglet.extlibs.earcut import zOrder

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
        V1_label_pos = vector + 0.6 * (UP + RIGHT)
        P1_label = MathTex(r"\begin{pmatrix} 2 \\ 1 \end{pmatrix}").move_to(V1_label_pos)
        V1_label = MathTex(r"\begin{bmatrix} 2 \\ 1 \end{bmatrix}").move_to(V1_label_pos)
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
        V2_label = MathTex(r"\begin{bmatrix} 1 \\ 3 \end{bmatrix}")
        # Arrange V1_label and V2_label side by side
        V1_label.save_state()
        V2_label.next_to(V1_label, RIGHT, buff=0.6)
        self.play(Create(V2))
        self.play(Create(V2_label))

        # Animate both labels to be side by side (if not already)
        self.play(
            V1_label.animate.move_to([-2, 2, 0]),
            V2_label.animate.next_to(V1_label, RIGHT, buff=0.6)
        )
        self.wait(0.5)

        # Create plus sign and insert between the labels
        plus_sign = MathTex(r"\mathbf{+}")
        plus_sign.next_to(V1_label, RIGHT, buff=0.08)
        self.play(Create(plus_sign))
        self.play(V2_label.animate.next_to(plus_sign, RIGHT, buff=0.08))
        rec.move_to(plus_sign.get_center())
        self.wait(1)

        # Move equation group to the right
        eqn_group = VGroup(rec, V1_label, plus_sign, V2_label)
        self.play(eqn_group.animate.shift(3 * RIGHT))
        self.wait(0.5)

        # Move yellow vector to tip of blue vector
        self.play(V2.animate.shift(V1.get_end()))
        self.wait(1)

        # Result vector and equals sign
        v3 = vector + vector_2
        V3 = Vector(v3, color=PURPLE)
        equals_sign = MathTex(r"\mathbf{=}").next_to(V2_label, RIGHT, buff=0.12)
        V3_label = MathTex(r"\begin{bmatrix} 3 \\ 4 \end{bmatrix}").move_to(V3.get_end() + 0.75*RIGHT + 2.2*DOWN).add_updater(lambda m: m.next_to(equals_sign, RIGHT, buff=0.12))
        rec.add_updater(lambda m: m.stretch_to_fit_width(w.get_value()))
        self.play(Create(equals_sign), Create(V3), Create(V3_label), rec.animate.next_to(equals_sign, LEFT, buff=-0.5), w.animate.set_value(5))
        self.wait(1)
        group_2 = VGroup(rec, V1_label, plus_sign, V2_label, equals_sign, V3_label)
        # Optional: move equation group to a fixed position
        self.play(group_2.animate.move_to([5.5, 5.5, 0]))
        self.play(group_2.animate.move_to([0, 5.5, 0]))
        self.wait(2.5)

        # Smooth scaling using ValueTracker
        scale_tracker = ValueTracker(1)
        V1.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, scale_tracker.get_value() * vector))
        V2.add_updater(lambda m: m.put_start_and_end_on(scale_tracker.get_value() * vector,
                                                        scale_tracker.get_value() * vector + vector_2))
        V3.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, scale_tracker.get_value() * vector + vector_2))

        # Visible scalar to the left of V1_label
        scalar_label = MathTex(r"\mathbf{1.0} \cdot").next_to(V1_label, LEFT, buff=0.15)
        scalar_label.add_updater(
            lambda m: m.become(
                MathTex(rf"\mathbf{{{scale_tracker.get_value():.1f}}} \cdot").next_to(V1_label, LEFT, buff=0.15)
            )
        )
        self.play(Create(scalar_label))

        # Continue with scaling animation
        self.play(scale_tracker.animate.set_value(2), run_time=2.5)
        self.wait(1.5)

        # Remove updaters after animation
        V1.clear_updaters()
        V2.clear_updaters()
        V3.clear_updaters()
        scalar_label.clear_updaters()
        scaled_2 = MathTex(r"\mathbf{2}").move_to(scalar_label.get_center())
        scaled_v3_label = MathTex(r"\begin{bmatrix} 5 \\ 5 \end{bmatrix}").move_to(V3_label.get_center())
        self.play(TransformMatchingTex(V3_label, scaled_v3_label), TransformMatchingTex(scalar_label, scaled_2))
        # Remove updaters after animation
        V1.clear_updaters()
        V2.clear_updaters()
        V3.clear_updaters()
        scalar_label.clear_updaters()
        V3_label.clear_updaters()
        # Group and fade out all elements
        self.wait(2.5)
        final_group = VGroup(*{V1, V2, V3, V1_label, V2_label, plus_sign, equals_sign, rec, scaled_v3_label, scaled_2})
        self.play(FadeOut(final_group), run_time=2)
        self.wait(2.5)

if __name__ == '__main__':
    scene = MyTwoDimensionalScene()
    scene.render()
    os.startfile(scene.renderer.file_writer.movie_file_path)