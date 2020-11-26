import numpy as np
import random
import glfw

from safety_gym.envs.engine import Engine
from mujoco_py import MjViewer, const

class SafetyEnv(Engine):
    def __init__(self, letters:str="aabbcddee", config={}):
        self.letters      = letters
        self.letter_types = list(set(letters))
        self.letter_types.sort()

        assert len(self.letter_types) < 9, 'We only support up to 8 letter types.'

        parent_config = {
            'robot_base': 'xmls/point.xml',
            'task': 'none',
            'observe_goal_lidar': False,
            'observe_box_lidar': False,
            'observe_hazards': True,
            'observe_vases': True,
            'constrain_hazards': True,
            'lidar_max_dist': 3,
            'lidar_num_bins': 16,
            'hazards_num': len(self.letter_types),
            'vases_num': 0,
            'num_steps': 10000
        }
        parent_config.update(config)

        super().__init__(parent_config)

    def set_hazards_cols(self):
        self.hazard_rgbs = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1]])

    def reward(self):
        robot_com = self.world.robot_com()

        # for h_pos in self.hazards_pos:
        #     h_dist = self.dist_xy(h_pos)
        #     if h_dist <= self.hazards_size:
        #         print("inside hazard")

        return super().reward()




class PlayEnv(SafetyEnv):
    def __init__(self, letters:str="aabbcddee"):
        config = {
            'observation_flatten': False
        }
        super().__init__(letters, config)

    def render(self, mode='human'):
        if self.viewer is None:
            self._old_render_mode = 'human'
            self.viewer = CustomViewer(self.sim)
            self.viewer.cam.fixedcamid = -1
            self.viewer.cam.type = const.CAMERA_FREE

            self.viewer.render_swap_callback = self.render_swap_callback
            # Turn all the geom groups on
            self.viewer.vopt.geomgroup[:] = 1
            self._old_render_mode = mode

        super().render()

    def obs(self):
        obs = super().obs()

        if not self.viewer is None:
            obs['key_pressed'] = self.viewer.consume_key()

        return obs


class CustomViewer(MjViewer):
    def __init__(self, sim):
        super().__init__(sim)
        self.key_pressed = None

    def consume_key(self):
        ret = self.key_pressed
        self.key_pressed = None

        return ret

    def key_callback(self, window, key, scancode, action, mods):
        self.key_pressed = key
        if action == glfw.RELEASE:
            self.key_pressed = -1

        super().key_callback(window, key, scancode, action, mods)
