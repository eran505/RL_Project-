import argparse
import datetime
import gym
import numpy as np
import itertools
from SAC.Agent import SAC
import torch
from torch.utils.tensorboard import SummaryWriter
from SAC.ReplayBuffer import ReplayBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'
print('Using device:', device)
# Additional Info when using cuda
if device == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')




parser = argparse.ArgumentParser()
parser.add_argument('--envId', default="LunarLanderContinuous-v2")
parser.add_argument('--policy', default="Gaussian")
parser.add_argument('--eval', type=bool, default=True)
parser.add_argument('--gamma', type=float, default=0.99,help='discount factor')
parser.add_argument('--tau', type=float, default=0.005,help='soft update')
parser.add_argument('--lr', type=float, default=0.0003,help='learning rate')
parser.add_argument('--alpha', type=float, default=0.2, help='Temperature parameter')
parser.add_argument('--test_episode',type=int,default=10)
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False,help='auto adjust alpha')
parser.add_argument('--seed', type=int, default=1254895)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_steps', type=int, default=1000001,help="maximum number of steps")
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--start_steps', type=int, default=2)
parser.add_argument('--target_update_interval', type=int, default=1)
parser.add_argument('--replay_size', type=int, default=100000)
parser.add_argument('--device', type=str, default=device)
args = parser.parse_args()




# Environment
env = gym.make(args.envId)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#Environment prop
input_shape=env.observation_space.shape[0]
n_actions=env.action_space


# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.envId,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
replay_memo = ReplayBuffer(args.replay_size,input_shape,n_actions.shape[0])

loss_info=None

# Training Loop
total_numsteps = 0
updates = 0
num_of_update=1
episode_idx=-1
while True:
    episode_idx+=1
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(replay_memo) > args.batch_size:
            # Number of updates per step in environment
            for i in range(num_of_update):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(replay_memo, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

                loss_info=[policy_loss,critic_2_loss]
        # if action >= 2 or action <= -2 or episode_idx>=90:
        #     print("action:", action)
        #     print("state:", state)

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        replay_memo.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, episode_idx)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(episode_idx, total_numsteps, episode_steps, round(episode_reward, 2)))
    print("Losses {}".format(loss_info))
    if episode_idx % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = args.test_episode
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, episode_idx)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()