from gym_minigrid.minigrid import *
from gym_minigrid.register import register

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from random import randint

class AdversarialEnv(MiniGridEnv):
    """
    An environment where a myopic agent will fail. The two possible goals are "Reach blue then green" or "Reach blue then red".
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.event_objs = []


        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate inner walls
        self.grid.vert_wall(4, 0)
        self.grid.horz_wall(4, 4)

        self.door_1 = Door(COLOR_NAMES[0], is_open=True)
        self.door_2 = Door(COLOR_NAMES[0], is_open=True)

        self.door_1_loc = (4,2)
        self.door_2_loc = (4,6)
        self.grid.set(*self.door_1_loc, self.door_1)
        self.grid.set(*self.door_2_loc, self.door_2)


        # Place a goal square in the bottom-right corner
        self.blue_goal_1_pos = (6, 8)
        self.blue_goal_2_pos = (6, 1)
        self.blue_goal_1 = Goal('blue')
        self.blue_goal_2 = Goal('blue')

        self.green_goal_pos = (8, 8)
        self.red_goal_pos = (8, 1)
        self.green_goal = Goal('green')
        self.red_goal = Goal('red')

        self.put_obj(self.green_goal, *self.green_goal_pos)
        self.put_obj(self.red_goal, *self.red_goal_pos)
        self.put_obj(self.blue_goal_1, *self.blue_goal_1_pos)
        self.put_obj(self.blue_goal_2, *self.blue_goal_2_pos)

        self.event_objs = []
        self.event_objs.append((self.blue_goal_1_pos, 'a'))
        self.event_objs.append((self.blue_goal_2_pos, 'a'))
        self.event_objs.append((self.green_goal_pos, 'b'))
        self.event_objs.append((self.red_goal_pos, 'c'))


        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent(top=(0,0), size=(3,height))


        # Green goal
        if randint(0,1) == 0:
        	self.mission = "reach the blue goal square then the green goal square"
        # Red goal
        else:
        	self.mission = "reach the blue goal square then the red goal square"
        

    def step(self, action):
        # Lock the door automatically behind you

        if action == self.actions.forward and self.agent_dir == 0:
            if tuple(self.agent_pos) == self.door_1_loc:
                self.door_1.is_open = False
                self.door_1.is_locked = True
            elif tuple(self.agent_pos) == self.door_2_loc:
                self.door_2.is_open = False
                self.door_2.is_locked = True

        obs, reward, done, _ = super().step(action)
        print(self.get_events())

        # if done and tuple(self.agent_pos) != self.target_pos:
        # 		reward = 0

        return obs, reward, False, {}

    def get_events(self):
        events = ""
        for obj in self.event_objs:
            if tuple(self.agent_pos) == obj[0]:
                events += obj[1]
        return events


class AdversarialEnv10x10(AdversarialEnv):
    def __init__(self):
        super().__init__(size=10, agent_start_pos=None)


register(

    id='MiniGrid-Adversarial-v0',
    entry_point='gym_minigrid.envs:AdversarialEnv10x10'
)
