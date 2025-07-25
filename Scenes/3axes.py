from manim import *
import numpy as np
import os
import time


v1 = np.array([1, 3, 1])
v2 = np.array([1, 1, 1])


l1 = r"\begin{bmatrix} 1 \ 2 \ 3 \end{bmatrix}"
l2 = r"\begin{bmatrix} 1 \ 1 \ 1 \end{bmatrix}"

class MyThreeDimensionalScene(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        V1 = Vector(v1, color=YELLOW)
        V2 = Vector(v2, color=RED)
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
        self.wait(2.5)


if __name__ == "__main__":
    t0 = time.perf_counter()
    scene = MyThreeDimensionalScene()
    scene.render()
    config.quality = 'low_quality'
    output_path = r'/media/videos/1080p60/MyThreeDimensionalScene.mp4'
    os.startfile(output_path)
    t1 = time.perf_counter()
    print(f"Scene rendered in {t1 - t0:.4f} seconds.")

