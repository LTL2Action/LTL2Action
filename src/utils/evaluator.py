import time
import torch
from torch_ac.utils.penv import ParallelEnv
import tensorboardX

import utils

"""
This class evaluates a model on a validation dataset generated online
via the sampler (ltl_sampler) that is passed in (model_name). It outputs
the results for visualization on TensorBoard by creating a folder under
the same directory as the trained model.
"""
class Eval:
    def __init__(self, env, model_name, ltl_sampler,
                seed=0, device="cpu", argmax=False,
                num_procs=1, ignoreLTL=False):

        self.device = device
        self.argmax = argmax
        self.num_procs = num_procs
        self.ignoreLTL = ignoreLTL

        self.model_dir = utils.get_model_dir(model_name)
        self.tb_writer = tensorboardX.SummaryWriter(self.model_dir + "/eval")

        # Load environments for evaluation
        eval_envs = []
        for i in range(self.num_procs):
            eval_envs.append(utils.make_env(env, ltl_sampler, seed))
        self.eval_envs = ParallelEnv(eval_envs)


    def eval(self, num_frames, episodes=100):
        # Load agent
        agent = utils.Agent(self.eval_envs.observation_space, self.eval_envs.action_space, self.model_dir + "/train", self.ignoreLTL,
                            device=self.device, argmax=self.argmax, num_envs=self.num_procs)


        # Run agent
        start_time = time.time()

        obss = self.eval_envs.reset()

        log_counter = self.num_procs
        log_episode_return = torch.zeros(self.num_procs, device=self.device)
        log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        # Initialize logs
        logs = {"num_frames_per_episode": [], "return_per_episode": []}
        while log_counter < episodes:
            actions = agent.get_actions(obss)
            obss, rewards, dones, _ = self.eval_envs.step(actions)
            agent.analyze_feedbacks(rewards, dones)

            log_episode_return += torch.tensor(rewards, device=self.device, dtype=torch.float)
            log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done in enumerate(dones):
                if done:
                    log_counter += 1
                    logs["return_per_episode"].append(log_episode_return[i].item())
                    logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

            mask = 1 - torch.tensor(dones, device=self.device, dtype=torch.float)
            log_episode_return *= mask
            log_episode_num_frames *= mask

        end_time = time.time()

        # Print logs
        num_frame_pe = sum(logs["num_frames_per_episode"])
        fps = num_frame_pe/(end_time - start_time)
        duration = int(end_time - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["frames", "FPS", "duration"]
        data   = [num_frame_pe, fps, duration]
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()

        # txt_logger.info(
        #     "F {:06} | FPS {:04.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
        #     .format(*data))

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        for field, value in zip(header, data):
                self.tb_writer.add_scalar(field, value, num_frames)
