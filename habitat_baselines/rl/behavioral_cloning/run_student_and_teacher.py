import os
import torch
import tqdm

from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.behavioral_cloning.agents import (
    D4Teacher,
    GaussianStudent,
)
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.config.default import get_config
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.utils import (
    batch_obs
)

from habitat import Config, logger

CHECKPOINT_PATH = '/coc/pskynet3/nyokoyama3/learnbycheat/exp2_sgd/checkpoints/6000_0.1655_0.0798.ckpt'

@baseline_registry.register_trainer(name="run_teacher")
class RunTeacher(BaseRLTrainer):
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        logger.info(f"env config: {config}")

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()
        
        self.config = config
        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))

        self.student = GaussianStudent(config.RL)
        self.student.actor_critic.load_state_dict(
             torch.load(CHECKPOINT_PATH)['state_dict']
        )
    def run(self):
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        test_recurrent_hidden_states = torch.zeros(
            self.student.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )

        # Student previous actions
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 2, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )

        number_of_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_episodes == -1:
            number_of_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_episodes:
                logger.warn(
                    f"Config specified {number_of_episodes} episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_episodes)
        self.student.actor_critic.eval()
        stats_episodes = dict()  # dict of dicts that stores stats per episode
        while (
            len(stats_episodes) < number_of_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.student.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )

            prev_actions.copy_(actions)

            step_actions = []
            for a in actions:
                a_tanh = torch.tanh(a)
                lin_vel, ang_vel = a_tanh[0].item(), a_tanh[1].item()
                d_action = min(
                    [(0,1.,0.), (1,-1.,0.), (2,1.,1.), (3,1.,-1.)],
                    key=lambda x:(lin_vel-x[1])**2+(ang_vel-x[2])**2
                )[0]
                print('d_action',d_action)
                step_actions.append(d_action)

            # outputs = self.envs.step([a[0].item() for a in actions])
            outputs = self.envs.step(step_actions)

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = dict()
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    # print('SPL:', infos[i]['spl'])
                    # print(infos[i])

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                torch.zeros(self.config.NUM_PROCESSES), # current_episode_reward
                prev_actions,
                batch,
                rgb_frames=[[] for _ in range(self.config.NUM_PROCESSES)],
            )
        self.envs.close()

if __name__ == '__main__':
    config = get_config(
        'habitat_baselines/config/pointnav/behavioral_cloning.yaml'
    )
    d = RunTeacher(config)
    d.run()