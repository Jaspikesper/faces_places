from manim import *
import numpy as np
from manim.utils.file_ops import open_file as open_media_file

np.random.seed(69)

class MyTwoDimensionalScene(Scene):
    def construct(self):
        self.camera.frame_width = 15
        self.camera.frame_height = 15
        number_plane = NumberPlane(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.6,
            }
        )
        dots = VGroup()
        population_mean = np.array([5, 2])
        population_covariance = np.array([[1, 0.25], [0.25, 1]])
        n = 25
        mv_normal_data = np.random.multivariate_normal(
            population_mean, population_covariance, n
        )
        for point in mv_normal_data:
            dot = Dot(np.append(point, 0), color=WHITE, radius=0.1)
            dots.add(dot)
        # Calculate and add center of gravity (mean) as a blue dot
        cog = np.mean(mv_normal_data, axis=0)
        cog_circle = Circle(color=RED, radius=0.15)
        cog_circle.move_to(np.append(cog, 0))
        self.play(Create(number_plane), run_time=2)
        self.play(Create(dots), run_time=2)
        self.play(FadeIn(cog_circle), run_time=2)
        self.wait(2.5)
        self.play(
            dots.animate.shift(-cog[0]* RIGHT - cog[1]*UP),
            cog_circle.animate.shift(-cog[0]* RIGHT - cog[1]*UP),
            run_time=2.5
        )
        self.wait(2.5)
        self.play(FadeOut(cog_circle))

if __name__ == '__main__':
    scene = MyTwoDimensionalScene()
    scene.render()
    open_media_file(scene.renderer.file_writer.movie_file_path)