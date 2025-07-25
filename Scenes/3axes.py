from manim import *
import numpy as np
import os
import time

v1 = np.array([1, 2, 3])
v2 = np.array([1, 1, 1])
projv1_v2 = v1 * np.dot(v2, v1) / np.dot(v1, v1)
negative_projection = -projv1_v2

l1 = r"\mathbf{v}_1"
l2 = r"\mathbf{v}_2"


class MyThreeDimensionalScene(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        V1 = Vector(v1, color=RED)
        V2 = Vector(v2)
        L1 = MathTex(l1)
        L2 = MathTex(l2)
        L1.next_to(V1.get_end(), RIGHT)
        L2.next_to(V2.get_end(), RIGHT)
        G1 = VGroup(V1, L1)
        G2 = VGroup(V2, L2)
        self.set_camera_orientation(
            phi=60 * DEGREES,
            theta=-90 * DEGREES,
        )
        self.play(Create(axes))
        self.play(Create(G1))
        self.play(Create(G2))
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(2.5)
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES, run_time=2.5)
        self.default_camera_orientation_kwargs = {'phi': 80 * DEGREES, 'theta': -30 * DEGREES, 'zoom': 1.5}
        self.wait(1)

        # Initial projection vector at origin
        PROJ_V1_V2 = Vector(projv1_v2, color=BLUE)
        PROJ_L1 = MathTex(r"\operatorname{proj}_{\mathbf{v}_1}\mathbf{v}_2", font_size=32)
        PROJ_L1.add_updater(lambda m: m.next_to(PROJ_V1_V2.get_end(), OUT + RIGHT + DOWN * 0.5))
        self.play(Create(PROJ_V1_V2), Create(PROJ_L1))
        self.wait(1.5)

        # New projection vector placed tip-to-tail at V2
        proj_shifted = Vector(projv1_v2, color=BLUE).shift(V2.get_end())
        proj_label_shifted = MathTex(r"\operatorname{proj}_{\mathbf{v}_1}\mathbf{v}_2", font_size=32)
        proj_label_shifted.add_updater(lambda m: m.next_to(proj_shifted.get_end(), OUT + RIGHT + DOWN * 0.5))

        # Animate movement and remove original
        self.play(
            ReplacementTransform(PROJ_V1_V2, proj_shifted),
            ReplacementTransform(PROJ_L1, proj_label_shifted)
        )
        self.wait(1.5)

        # Negative projection vector at tip of v2
        NEG = Vector(negative_projection, color=YELLOW).shift(V2.get_end())
        NEG_L1 = MathTex(r"-\operatorname{proj}_{\mathbf{v}_1}\mathbf{v}_2")
        NEG_L1.add_updater(lambda m: m.next_to(NEG.get_end(), OUT * 2 + RIGHT + DOWN * 0.7))

        self.play(
            ReplacementTransform(proj_shifted, NEG),
            ReplacementTransform(proj_label_shifted, NEG_L1)
        )

        solution = (v2 + negative_projection) * 5 # Scale up the solution vector for visibility
        SOLUTION = Vector(solution, color=RED)

        a = solution * 0.1
        b = v1 * 0.1
        c = a + b
        line_1 = Line(a, c, color=WHITE)
        line_2 = Line(b, c, color=WHITE)
        print(f"v1: {v1}, v2: {v2}")
        print(f"solution: {solution}")
        print(f"angle between v1 and solution: {np.degrees(np.arccos(np.dot(v1, solution) / (np.linalg.norm(v1) * np.linalg.norm(solution)))):.2f} degrees")
        self.wait(1.5)

        self.play(ReplacementTransform(V2, SOLUTION), FadeOut(NEG), FadeOut(NEG_L1))
        self.wait(1.5)

        self.set_camera_orientation(phi=45 * DEGREES, theta=-160 * DEGREES, run_time=2.5)
        self.begin_ambient_camera_rotation()
        self.wait(1)

        #group up everything and scale up
        final_group = VGroup(axes, G1, G2, SOLUTION)
        self.play(final_group.animate.scale(2.5))
        self.wait(1)
        self.play(FadeIn(line_1), FadeIn(line_2))
        self.wait(1.5)

if __name__ == "__main__":
    t0 = time.perf_counter()
    scene = MyThreeDimensionalScene()
    scene.render()
    config.quality = 'low_quality'
    output_path = r'C:\Users\jaspe\PycharmProjects\PythonProject8\Scenes\media\videos\1080p60\MyThreeDimensionalScene.mp4'
    os.startfile(output_path)
    t1 = time.perf_counter()
    print(f"Scene rendered in {t1 - t0:.4f} seconds.")
