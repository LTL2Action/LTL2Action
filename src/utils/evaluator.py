import time
import torch
from torch_ac.utils.penv import ParallelEnv
#import tensorboardX

import utils
import argparse
import datetime
from envs.gym_letters.letter_env import LetterEnv

"""
This class evaluates a model on a validation dataset generated online
via the sampler (ltl_sampler) that is passed in (model_name). It outputs
the results for visualization on TensorBoard by creating a folder under
the same directory as the trained model.
"""
class Eval:
    def __init__(self, env, model_name, ltl_sampler,
                seed=0, device="cpu", argmax=False,
                num_procs=1, ignoreLTL=False, progression_mode=True, gnn=None, recurrence=1, dumb_ac = False, discount=0.99):

        self.env = env
        self.device = device
        self.argmax = argmax
        self.num_procs = num_procs
        self.ignoreLTL = ignoreLTL
        self.progression_mode = progression_mode
        self.gnn = gnn
        self.recurrence = recurrence
        self.dumb_ac = dumb_ac
        self.discount = discount

        self.model_dir = utils.get_model_dir(model_name, storage_dir="")
        #self.tb_writer = tensorboardX.SummaryWriter(self.model_dir + "/eval-" + ltl_sampler)

        # Load environments for evaluation
        eval_envs = []
        for i in range(self.num_procs):
            eval_envs.append(utils.make_env(env, progression_mode, ltl_sampler, seed, 0, False))

        eval_envs[0].reset()
        if isinstance(eval_envs[0].env, LetterEnv):
            for env in eval_envs:
                env.env.map = eval_envs[0].env.map

        self.eval_envs = ParallelEnv(eval_envs)




    def eval(self, num_frames, episodes=100, stdout=True):
        # Load agent

        agent = utils.Agent(self.eval_envs.envs[0], self.eval_envs.observation_space, self.eval_envs.action_space, self.model_dir + "/train", 
                self.ignoreLTL, self.progression_mode, self.gnn, recurrence = self.recurrence, dumb_ac = self.dumb_ac, device=self.device, argmax=self.argmax, num_envs=self.num_procs)


        # Run agent
        start_time = time.time()

        obss = self.eval_envs.reset()

        # This is only giving unbiased evaluations for n_procs = 1 right now
        log_counter = 0
        assert(self.num_procs == 1)

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


        return logs["return_per_episode"], logs["num_frames_per_episode"]

        # Print logs
        # num_frame_pe = sum(logs["num_frames_per_episode"])
        # fps = num_frame_pe/(end_time - start_time)
        # duration = int(end_time - start_time)
        # return_per_episode = utils.synthesize(logs["return_per_episode"])
        # num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
        # average_discounted_return = utils.average_discounted_return(logs["return_per_episode"], logs["num_frames_per_episode"], self.discount)

        # header = ["frames", "FPS", "duration"]
        # data   = [num_frame_pe, fps, duration]
        # header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        # data += num_frames_per_episode.values()
        # header += ["average_discounted_return"]
        # data += [average_discounted_return]

        # # txt_logger.info(
        # #     "F {:06} | FPS {:04.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
        # #     .format(*data))

        # header += ["return_" + key for key in return_per_episode.keys()]
        # data += return_per_episode.values()

        # for field, value in zip(header, data):
        #     if stdout:
        #         print(field, value)
            #else:
                #self.tb_writer.add_scalar(field, value, num_frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ltl-sampler", default="Default",
                    help="the ltl formula template to sample from (default: DefaultSampler)")
    parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
    parser.add_argument("--model-paths", required=True, nargs="+",
                    help="path of the model, or a regular expression")
    parser.add_argument("--procs", type=int, default=1,
                    help="number of processes (default: 1)")
    parser.add_argument("--eval-episodes", type=int,  default=5,
                    help="number of episodes to evaluate on (default: 5)")
    parser.add_argument("--env", default="Letter-7x7-v3",
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")

    parser.add_argument("--ignoreLTL", action="store_true", default=False,
                    help="the network ignores the LTL input")
    parser.add_argument("--progression-mode", default="full",
                    help="Full: uses LTL progression; partial: shows the propositions which progress or falsify the formula; none: only original formula is seen. ")
    parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--gnn", default="RGCN_8x32_ROOT_SHARED", help="use gnn to model the LTL (only if ignoreLTL==True)")


    args = parser.parse_args()

    print(args)


    # envs = []

    # for i in range(args.procs):
    #     envs.append(utils.make_env(args.env, args.progression_mode, args.ltl_sampler, args.seed, args.int_reward, args.noLTL))
    logs_returns_per_episode = []
    logs_num_frames_per_episode = [] 



    for model_path in args.model_paths:
        idx = model_path.find("seed:") + 5
        seed = int(model_path[idx:idx+2].strip("_"))

        eval = utils.Eval(args.env, model_path, args.ltl_sampler,
                     seed=seed, device=torch.device("cpu"), argmax=False,
                     num_procs=args.procs, ignoreLTL=args.ignoreLTL, progression_mode=args.progression_mode, gnn=args.gnn, recurrence=args.recurrence, dumb_ac=False, discount=args.discount)
        rpe, nfpe = eval.eval(-1, episodes=args.eval_episodes, stdout=True)
        logs_returns_per_episode += rpe
        logs_num_frames_per_episode += nfpe 

        print(sum(rpe), seed, model_path)


    num_frame_pe = sum(logs_num_frames_per_episode)
    return_per_episode = utils.synthesize(logs_returns_per_episode)
    num_frames_per_episode = utils.synthesize(logs_num_frames_per_episode)
    average_discounted_return, error = utils.average_discounted_return(logs_returns_per_episode, logs_num_frames_per_episode, args.discount, include_error=True)

    header = ["frames"]
    data   = [num_frame_pe]
    header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
    data += num_frames_per_episode.values()
    header += ["average_discounted_return", "err"]
    data += [average_discounted_return, error]

    # txt_logger.info(
    #     "F {:06} | FPS {:04.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
    #     .format(*data))

    header += ["return_" + key for key in return_per_episode.keys()]
    data += return_per_episode.values()

    for field, value in zip(header, data):
        print(field, value)


    # eval_env = args.envs
    # eval_sampler = args.ltl_sampler
    # model_name = args.model_name # format: "{eval_env}_{eval_sampler}_ppo_seed{args.seed}_{date}"
    # device = "cpu"
    # eval_procs = args.procs
    # ignoreLTL = args.ignoreLTL
    # eval_episodes = args.eval_episodes


    # eval = utils.Eval(eval_env, model_name, eval_sampler,
    #             seed=args.seed, device=device, num_procs=eval_procs, ignoreLTL=args.ignoreLTL, gnn=args.gnn)
    # eval.eval(-1, episodes=args.eval_episodes, stdout=True)

