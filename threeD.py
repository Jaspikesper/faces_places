from manim import *
import numpy as np

v1 = np.array([0.7, 3, 3.5])
v2 = np.array([1, 1, 1])


class MyThreeDimensionalScene(VectorScene):
    def construct(self):
        V1 = Vector(v1, color=YELLOW)
        V2 = Vector(v2, color=RED)
        L1 = V1.coordinate_label()
        L2 = V2.coordinate_label()
        self.play(
            Succession(
                Create(NumberPlane()),
                Create(V1),
                Create(L1),
                Create(V2),
                Create(L2),
                Wait(2.5)
            )
        )
        self.wait(2.5)
