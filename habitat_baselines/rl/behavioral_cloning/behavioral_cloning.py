import os
import tqdm
import torch
import torch.optim as optim
from collections import defaultdict

from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.behavioral_cloning.agents import (
    D4Teacher,
    GaussianStudent
)
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.config.default import get_config
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.utils import batch_obs

from habitat import Config, logger

BATCH_LENGTH = 8
CHECKPOINT_PATH = '/coc/pskynet3/nyokoyama3/learnbycheat/exp2_sgd/checkpoints/6000_0.1655_0.0798.ckpt'

def discrete_to_continuous(actions, device):
    continuous_actions = []
    for a in actions:
        if a[0].item() == 0: # STOP
            continuous_actions.append([1.,0.])
        elif a[0].item() == 1: # FORWARD
            continuous_actions.append([-1.,0.])
        elif a[0].item() == 2: # LEFT
            continuous_actions.append([1.,1.])
        elif a[0].item() == 3: # RIGHT
            continuous_actions.append([1.,-1.])
    return torch.tensor(continuous_actions, device=device)
def copy_batch(batch, device):
    batch_copy = defaultdict(list)
    for sensor in batch:
        batch_copy[sensor] = batch[sensor].detach().clone()
    return batch_copy
def cat_batch(batch_list, device):
    batch = defaultdict(list)

    for b in batch_list:
        for sensor in b:
            batch[sensor].append(b[sensor])

    for sensor in batch:
        batch[sensor] = torch.cat(batch[sensor], dim=0)
    return batch

@baseline_registry.register_trainer(name="behavioral_cloning")
class BehavioralCloning(BaseRLTrainer):
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        logger.info(f"env config: {config}")

        # Faster loading
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = 'val'
        config.freeze()

        self.config = config
        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))
        self.teacher = D4Teacher(config.RL)
        self.student = GaussianStudent(config.RL)

        self.student.actor_critic.load_state_dict(
             torch.load(CHECKPOINT_PATH)['state_dict']
        )

    def train(self):
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        teacher_hidden_states = torch.zeros(
            self.teacher.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )
        teacher_prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )

        self.teacher.actor_critic.eval()
        
        # self.optimizer = optim.Adam(
        #     list(filter(
        #         lambda p: p.requires_grad,
        #         self.student.actor_critic.parameters()
        #     )),
        #     lr=2.5e-4,
        #     eps=1e-5,
        # )
        self.optimizer = optim.SGD(
            list(filter(
                lambda p: p.requires_grad,
                self.student.actor_critic.parameters()
            )),
            lr=0.01,
            momentum=0.8,
            weight_decay=5e-4
        )
        batch_num = 0
        while True:
            current_episodes = self.envs.current_episodes()

            obs_buff = []
            prev_actions_buff = []
            hidden_state_buff = []
            not_done_buff = []
            teacher_labels = []
            hidden_state_labels = []

            '''
            Student gets a batch, hs, prev_a, and not_done
            hstack-ing batch is pretty straightforward
            hidden states and prev_a is hard
            prev_a should be stolen from the teacher
            '''

            for step in range(BATCH_LENGTH):
                obs_buff.append(
                    copy_batch(batch, device=self.device)
                )
                hidden_state_buff.append(
                    (
                        teacher_hidden_states
                        .detach()
                        .clone()
                        # .to(torch.device('cpu'))
                    )
                )
                not_done_buff.append(
                    (
                        not_done_masks
                        .detach()
                        .clone()
                        # .to(torch.device('cpu'))
                    )
                )

                # Must remap actions
                prev_actions_buff.append(
                    discrete_to_continuous(
                        teacher_prev_actions,
                        device=self.device
                    )
                )

                with torch.no_grad():
                    (
                        _,
                        teacher_actions,
                        _,
                        teacher_hidden_states,
                    ) = self.teacher.actor_critic.act(
                        batch,
                        teacher_hidden_states,
                        teacher_prev_actions,
                        not_done_masks,
                        deterministic=True,
                    )

                teacher_labels.append(
                    discrete_to_continuous(
                        teacher_actions,
                        device=self.device
                    )
                )
                hidden_state_labels.append(
                    (
                        teacher_hidden_states
                        .detach()
                        .clone()
                        # .to(torch.device('cpu'))
                    )
                )

                teacher_prev_actions.copy_(teacher_actions)

                outputs = self.envs.step([a[0].item() for a in teacher_actions])

                observations, rewards, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]

                batch = batch_obs(observations, device=self.device)

                not_done_masks = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
                    dtype=torch.float,
                    device=self.device,
                )

            # Student inputs
            obs_buff = cat_batch(obs_buff, device=self.device)
            hidden_state_buff = torch.cat(hidden_state_buff, dim=1)
            prev_actions_buff = torch.cat(prev_actions_buff, dim=0)
            not_done_buff = torch.cat(not_done_buff, dim=0)
            (
                _,
                student_actions,
                _,
                hidden_state_out,
            ) = self.student.actor_critic.act(
                obs_buff,
                hidden_state_buff,
                prev_actions_buff,
                not_done_buff,
                deterministic=True,
            )
            student_actions = torch.tanh(student_actions)
            teacher_labels = torch.cat(teacher_labels, dim=0)
            print(
                'student_actions',
                student_actions[-1].cpu().detach().numpy(),
                'teacher_labels',
                teacher_labels[-1].cpu().detach().numpy(),
                'teacher_actions',
                teacher_actions[-1].cpu().detach().numpy(),
            )
            hidden_state_labels = torch.cat(hidden_state_labels, dim=1)
            action_loss = (student_actions - teacher_labels).pow(2).mean()
            hidden_state_loss = (
                hidden_state_out - hidden_state_labels
            ).pow(2).mean()

            self.optimizer.zero_grad()
            total_loss = (
                action_loss
                + hidden_state_loss * 0.1
            )

            total_loss.backward()

            self.optimizer.step()

            batch_num += 1

            print(
                f'#{batch_num}'
                f'  total: {total_loss.item():.4f}'
                f'  action: {action_loss.item():.4f}'
                f'  hidden_state: {hidden_state_loss.item():.4f}'
            )

            if batch_num % 200 == 0:
                checkpoint = {
                    "state_dict": self.student.actor_critic.state_dict(),
                    "config": self.config,
                }

                file_name = (
                    f'2_{batch_num:03d}_'
                    f'{action_loss.item():.4f}_'
                    f'{hidden_state_loss.item():.4f}'
                    '.ckpt'
                )
                if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
                    os.makedirs(self.config.CHECKPOINT_FOLDER)
                torch.save(
                    checkpoint,
                    os.path.join(
                        self.config.CHECKPOINT_FOLDER,
                        file_name
                    )
                )

        self.envs.close()

if __name__ == '__main__':
    config = get_config(
        'habitat_baselines/config/pointnav/behavioral_cloning.yaml'
    )
    d = BehavioralCloning(config)
    d.train()