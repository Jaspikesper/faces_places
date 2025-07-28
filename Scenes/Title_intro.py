from manim import *

class TitleIntro(Scene):
    title = "Face the Facts: A PCA and Dimensionality Reduction Demonstration in Manim"
    quote = "Linear algebra has moved to the center of machine learning, and we need to be there."
    author = "-Gil Strang"
    def construct(self):
        T = Title(title, font_size=48, color=WHITE)
        self.add(T)
        Q = Text(self.quote, font_size=24, color=WHITE).next_to(T, DOWN, buff=0.5)
        A = Text(self.author, font_size=20, color=WHITE).next_to(Q, DOWN, buff=0.2)
        self.play(Create(Q), run_time=2)
        self.play(Create(A))


if __name__ == '__main__':
    config.quality = 'low_quality'
    scene = TitleIntro()
    scene.render()
    os.startfile(scene.renderer.file_writer.movie_file_path)