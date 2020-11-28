from gym_minigrid.minigrid import *
from gym_minigrid.register import register

## Colored Goals 
class CGoal(WorldObj):
    def __init__(self, color):
        super().__init__('goal', color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])