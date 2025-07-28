from manim import *
import pandas as pd

class TitleIntro(Scene):
    df = pd.DataFrame
    columns = ['X', 'Y']
    title = "Taking the Mean of Multidimensional Data"
    data = [np.random.randint(0, 10, size=2) for _ in range(25)]

    def construct(self):
        t = Title(title, font_size=48, color=WHITE)
        self.add(t)


if __name__ == '__main__':
    config.quality = 'low_quality'
    scene = TitleIntro()
    scene.render()
    os.startfile(scene.renderer.file_writer.movie_file_path)