import gym
import glfw
from mujoco_py import MjViewer, const

"""
A simple wrapper for SafetyGym envs. It uses the PlayViewer that listens to key_pressed events
and passes the id of the pressed key as part of the observation to the agent.
(used to control the agent via keyboard)

Should NOT be used for training!
"""
class Play(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    # Shows a text on the upper right corner of the screen (currently used to display the LTL formula)
    def show_text(self, text):
        self.env.viewer.show_text(text)

    def render(self, mode='human'):
        if self.env.viewer is None:
            self.env._old_render_mode = 'human'
            self.env.viewer = PlayViewer(self.env.sim)
            self.env.viewer.cam.fixedcamid = -1
            self.env.viewer.cam.type = const.CAMERA_FREE

            self.env.viewer.render_swap_callback = self.env.render_swap_callback
            # Turn all the geom groups on
            self.env.viewer.vopt.geomgroup[:] = 1
            self.env._old_render_mode = mode

        super().render()

    def wrap_obs(self, obs):
        if not self.env.viewer is None:
            obs['key_pressed'] = self.env.viewer.consume_key()

        return obs

    def reset(self):
        obs = self.env.reset()

        return self.wrap_obs(obs)

    def step(self, action):
        next_obs, original_reward, env_done, info = self.env.step(action)

        return self.wrap_obs(next_obs), original_reward, env_done, info


class PlayViewer(MjViewer):
    def __init__(self, sim):
        super().__init__(sim)
        self.key_pressed = None
        self.custom_text = None

    def show_text(self, text):
        self.custom_text = text

    def consume_key(self):
        ret = self.key_pressed
        self.key_pressed = None

        return ret

    def key_callback(self, window, key, scancode, action, mods):
        self.key_pressed = key
        if action == glfw.RELEASE:
            self.key_pressed = -1

        super().key_callback(window, key, scancode, action, mods)

    def _create_full_overlay(self):
        if (self.custom_text): self.add_overlay(const.GRID_TOPRIGHT, self.custom_text, "")
        super()._create_full_overlay()
