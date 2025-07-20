import manim
from manim import Scene, ORIGIN
from PsiCreature.PsiCreature.psi_creature import PsiCreature
class ShowPsiCreature(Scene):
    def construct(self):
        # Create a PsiCreature object
        creature = PsiCreature()
        # Optionally, move or scale the creature
        creature.move_to(ORIGIN).scale(2)
        # Add it to the scene
        self.add(creature)
        # Wait for 2 seconds
        self.wait(2)