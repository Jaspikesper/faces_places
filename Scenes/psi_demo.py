from manim import *
from PsiCreature.PsiCreature.psi_creature import PsiCreature
import os

class PsiDemo(Scene):
    def construct(self):
        # 1. Initialize psi_creature at (-20, -20) with zero opacity
        psi = PsiCreature().set_opacity(0)
        self.play(psi.change_state("pondering"))

        self.play(psi.look_at([1, 1, 0]))
        psi.move_to([-20, -20, 0])
        # 2. Change its state to pondering
        self.wait(5)
        # 3. Move to (-3, -2) while simultaneously fading in
        self.play(
            psi.animate.set_opacity(1).move_to([-1, -1, 0]),
            run_time=2
        )
        self.wait(1)
        self.play(psi.blink())
        self.wait(2.5)
        self.play(psi.change_state("default"))
        self.wait(2.5)


if __name__ == '__main__':
    config.quality = 'low_quality'
    scene = PsiDemo()
    scene.render()
    os.startfile(scene.renderer.file_writer.movie_file_path)