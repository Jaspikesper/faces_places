from manim import *

class SingleVectorScene(Scene):
    def construct(self):
        # Create a vector from the origin to (2, 1)
        vector = Vector([2, 1], color="#FFD700")
        vec_silver = Vector([-1, 2], color="#C0C0C0")
        vector.set_sheen(0.5, direction=LEFT)
        vec_silver.set_sheen(0.5, direction=UP)
        self.add(vector)
        self.add(vec_silver)
        self.wait(1)

if __name__ == "__main__":
    from manim import config, tempconfig
    import sys

    # Optional: set the output file name and other config
    with tempconfig({"quality": "medium_quality", "preview": True}):
        scene = SingleVectorScene()
        scene.render()