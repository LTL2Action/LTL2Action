if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../../')

import random, math, os
import numpy as np
import gym
from gym import spaces

class LetterEnv(gym.Env):
    """
    This environment is a grid with randomly located letters on it
    We ensure that there is a clean path to any of the letters (a path that includes no passing by any letter)
    Note that steping outside the map causes the agent to appear on the other extreme
    """

    def __init__(self, grid_size:int, letters:str, use_fixed_map:float, use_agent_centric_view:float, timeout:int):
        """
            grid_size:
                - (int) size of the grid
            letters:
                - (str) letters that the grid will include in random locations (there could be repeated letters)
            use_fixed_map:
                - (bool) if True, then the map will be fixed for the whole training set
            timeout:
                - (int) maximum lenght of the episode
        """
        assert not use_agent_centric_view or grid_size%2==1, "Agent-centric view is only available for odd grid-sizes"
        self.grid_size     = grid_size
        self.letters       = letters
        self.use_fixed_map = use_fixed_map
        self.use_agent_centric_view = use_agent_centric_view
        self.letter_types = list(set(letters))
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size,grid_size,len(self.letter_types)+1), dtype=np.uint8)
        self.num_episodes = 0
        self.time = 0
        self.timeout = timeout
        self.map   = None
        self.agent = (0,0)
        self.locations = [(i,j) for i in range(grid_size) for j in range(grid_size) if (i,j) != (0,0)]
        self.actions = [(-1,0),(1,0),(0,-1),(0,1)]


    def step(self, action):
        """
        This function executes an action in the environment
        """
        di,dj = self.actions[action]
        agent_i = (self.agent[0] + di + self.grid_size) % self.grid_size
        agent_j = (self.agent[1] + dj + self.grid_size) % self.grid_size
        self.agent = agent_i,agent_j
        self.time += 1
        reward = 0.0
        done = self.time > self.timeout
        obs = self._get_observation()

        return obs, reward, done, {}

    def _get_observation(self):
        obs = np.zeros(shape=(self.grid_size,self.grid_size,len(self.letter_types)+1),dtype=np.uint8)

        # Getting agent-centric view (if needed)
        c_map, agent = self.map, self.agent
        if self.use_agent_centric_view:
            c_map, agent = self._get_centric_map()

        # adding objects
        for loc in c_map:
            letter_id = self.letter_types.index(c_map[loc])
            obs[loc[0],loc[1],letter_id] = 1

        # adding agent
        obs[agent[0],agent[1],len(self.letter_types)] = 1
        return obs

    def seed(self, seed=None):
        if (self.use_fixed_map): random.seed(seed)

    def reset(self):
        """
        This function resets the world and collects the first observation.
        """
        if not self.use_fixed_map:
            self.map = None

        # Sampling a new map
        while self.map is None:
            # Sampling a random map
            self.map = {}
            random.shuffle(self.locations)
            for i in range(len(self.letters)):
                self.map[self.locations[i]] = self.letters[i]
            # Checking that the map is valid
            if _is_valid_map(self.map, self.grid_size, self.actions):
                break
            self.map = None

        # Locating the agent into (0,0)
        self.agent = (0,0)

        # Aux values
        self.time = 0
        self.num_episodes += 1
        obs = self._get_observation()

        return obs

    def _get_centric_map(self):
        center = self.grid_size//2
        agent  = (center,center)
        delta  = center - self.agent[0], center - self.agent[1]
        c_map  = {}
        for loc in self.map:
            new_loc_i = (loc[0] + delta[0] + self.grid_size) % self.grid_size
            new_loc_j = (loc[1] + delta[1] + self.grid_size) % self.grid_size
            c_map[(new_loc_i,new_loc_j)] = self.map[loc]
        return c_map, agent

    def show(self):
        c_map, agent = self.map, self.agent
        if self.use_agent_centric_view:
            c_map, agent = self._get_centric_map()
        print("*"*(self.grid_size+2))
        for i in range(self.grid_size):
            line = "*"
            for j in range(self.grid_size):
                if (i,j) == agent:
                    line += "A"
                elif (i,j) in c_map:
                    line += c_map[(i,j)]
                else:
                    line += " "
            print(line+"*")
        print("*"*(self.grid_size+2))
        print("Events:", self.get_events(), "\tTimeout:", self.timeout - self.time)

    def show_features(self):
        obs = self._get_observation()
        print("*"*(self.grid_size+2))
        for i in range(self.grid_size):
            line = "*"
            for j in range(self.grid_size):
                if np.amax(obs[i,j,:]) > 0:
                    line += str(np.argmax(obs[i,j,:]))
                else:
                    line += " "
            print(line+"*")
        print("*"*(self.grid_size+2))


    def get_events(self):
        if self.agent in self.map:
            return self.map[self.agent]
        return ""

    def get_propositions(self):
        return self.letter_types

def _is_valid_map(map, grid_size, actions):
    open_list = [(0,0)]
    closed_list = set()
    while open_list:
        s = open_list.pop()
        closed_list.add(s)
        if s not in map:
            for di,dj in actions:
                si = (s[0] + di + grid_size) % grid_size
                sj = (s[1] + dj + grid_size) % grid_size
                if (si,sj) not in closed_list and (si,sj) not in open_list:
                    open_list.append((si,sj))
    return len(closed_list) == grid_size*grid_size



class LetterEnv4x4(LetterEnv):
    def __init__(self):
        super().__init__(grid_size=4, letters="aabbcddee", use_fixed_map=False, use_agent_centric_view=False, timeout=100)


class LetterEnvFixedMap4x4(LetterEnv):
    def __init__(self):
        super().__init__(grid_size=4, letters="aabbcddee", use_fixed_map=True, use_agent_centric_view=False, timeout=100)


class LetterEnv5x5(LetterEnv):
    def __init__(self):
        super().__init__(grid_size=5, letters="aabbccddee", use_fixed_map=False, use_agent_centric_view=False, timeout=150)


class LetterEnvFixedMap5x5(LetterEnv):
    def __init__(self):
        super().__init__(grid_size=5, letters="aabbccddee", use_fixed_map=True, use_agent_centric_view=False, timeout=150)


class LetterEnvAgentCentric5x5(LetterEnv):
    def __init__(self):
        super().__init__(grid_size=5, letters="aabbccddee", use_fixed_map=False, use_agent_centric_view=True, timeout=150)

class LetterEnvShortAgentCentric5x5(LetterEnv):
    def __init__(self):
        super().__init__(grid_size=5, letters="aabbccddee", use_fixed_map=False, use_agent_centric_view=True, timeout=20)

class LetterEnvAgentCentricFixedMap5x5(LetterEnv):
    def __init__(self):
        super().__init__(grid_size=5, letters="aabbccddee", use_fixed_map=True, use_agent_centric_view=True, timeout=150)

class LetterEnvShortAgentCentricFixedMap5x5(LetterEnv):
    def __init__(self):
        super().__init__(grid_size=5, letters="aabbccddee", use_fixed_map=True, use_agent_centric_view=True, timeout=20)

class LetterEnv7x7(LetterEnv):
    def __init__(self):
        super().__init__(grid_size=7, letters="aabbccddeeffgghhiijjkkll", use_fixed_map=False, use_agent_centric_view=False, timeout=500)

class LetterEnvFixedMap7x7(LetterEnv):
    def __init__(self):
        super().__init__(grid_size=7, letters="aabbccddeeffgghhiijjkkll", use_fixed_map=True, use_agent_centric_view=False, timeout=500)


class LetterEnvAgentCentric7x7(LetterEnv):
    def __init__(self):
        super().__init__(grid_size=7, letters="aabbccddeeffgghhiijjkkll", use_fixed_map=False, use_agent_centric_view=True, timeout=500)

class LetterEnvAgentCentricFixedMap7x7(LetterEnv):
    def __init__(self):
        super().__init__(grid_size=7, letters="aabbccddeeffgghhiijjkkll", use_fixed_map=True, use_agent_centric_view=True, timeout=500)



# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    # commands
    str_to_action = {"w":0,"s":1,"a":2,"d":3}
    grid_size = 5
    letters   = "aabbccddee"
    use_fixed_map = False
    timeout = 10
    use_agent_centric_view = False

    # play the game!
    game = LetterEnv(grid_size, letters, use_fixed_map, use_agent_centric_view, timeout)
    while True:
        # Episode
        game.reset()
        while True:
            game.show()
            game.show_features()
            print("\nAction? ", end="")
            a = input()
            print()
            # Executing action
            if a in str_to_action:
                obs, reward, done, _ = game.step(str_to_action[a])
                if done:
                    break
            else:
                print("Forbidden action")
        game.show()
