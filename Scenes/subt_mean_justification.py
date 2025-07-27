import numpy as np
from manim import *
import os


my_array = np.array([[2, 1], [1, 1], [0, 1], [3, 3], [-1, 1]])
my_array = my_array.astype(str)

class Tab(Scene):
    def construct(self):
        # Create a table with the array
        table = Table(
            my_array
        )
        self.play(Create(table))
        self.wait(2.5)

