import gym
import cv2
import torch
import datetime
import argparse
import argcomplete
import numpy as np
from tqdm import tqdm

from ddpg import DDPGAgent

gym.envs.register(
    id='CarRacing-v1',
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=10000,
)

RESULTS_EVERY_N_EPISODES = 1
LEARNING_RATE = 1e-3
REPLAY_MEMORY_SIZE = 25000
MINIBATCH_SIZE = 64
GAMMA = 0.95
UPDATE_TARGET_EVERY = 100
EPSILON = 1
TAU = 0.01

torch.device('cuda')

class Agent:
    def __init__(self):
        self.env_name = None
        self.env = None
        self.episodes = None
        self.epsilon = EPSILON
        self.results_every_n_episodes = RESULTS_EVERY_N_EPISODES

    def parser(self):
        argparser = argparse.ArgumentParser()
        argparser.add_argument(
            '-e',
            '--env',
            type=str,
            default='CarRacing-v1',
            help='OpenAI Gym environment'
        )
        argparser.add_argument(
            '-ep',
            '--episodes',
            type=int,
            default=200,
            help='Number of episodes to run'
        )
        argparser.add_argument(
            '--play',
            type=bool,
            default=False,
            help='Play using pre-trained models'
        )
        argcomplete.autocomplete(argparser)
        args = argparser.parse_args()

        return args

    def initialize_env(self, args):
        self.env_name = args.env
        self.env = gym.make(args.env)
        self.episodes = args.episodes
        self.epsilon_decay_value = 3e-6

    def convert_gray(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = state.reshape(1, *state.shape)/255
        return state

    def learn(self):
        agent = DDPGAgent(
            env=self.env,
            replay_memory_size=REPLAY_MEMORY_SIZE,
            learning_rate=LEARNING_RATE,
            batch_size=MINIBATCH_SIZE,
            gamma=GAMMA,
            tau=TAU
        )

        stats = {'scores': [], 'avg': [], 'min': [], 'max': []}
        for ep in tqdm(range(1, self.episodes + 1), ascii=True, unit='episodes'):

            print(self.epsilon)
            action_stats = [0, 0]
            current_state = self.env.reset()
            current_state = self.convert_gray(current_state)

            done = False
            score = 0
            steps = 0

            while not done:
                steps += 1

                if np.random.random() > self.epsilon:
                    action_stats[0] += 1
                    action = agent.get_action(current_state)
                else:
                    action_stats[1] += 1
                    action = self.env.action_space.sample()
                    action[2] = min(action[2], 0.2)
                    action[1] = action[1]*2

                new_state, reward, done, _ = self.env.step(action)
                if ep % self.results_every_n_episodes == 0:
                    self.env.render()

                score += reward

                new_state = self.convert_gray(new_state)

                agent.memory.push(current_state, action, reward, new_state)

                if steps % 64 == 0:
                    agent.update()

                current_state = new_state

                if self.epsilon > 0.1:
                    self.epsilon -= self.epsilon_decay_value

                if score < 0:
                    break

            print(action_stats)
            print(score)
            stats['scores'].append(score)
        self.env.close()
        return agent.actor

    def save_model(self, model):
        model.save_weights(
            "./models/{}_{}.model".format(self.env_name, str(datetime.datetime.now())))
        return

    def load_model(self):
        model = None
        return model

    def play(self):
        model = self.load_model()
        actions = {}
        for _ in range(3):
            current_state = self.env.reset()
            current_state = self.convert_gray(current_state)
            done = False
            score = 0
            while not done:
                action = model.predict(
                    np.array(current_state).reshape(-1, *current_state.shape)/255)[0]

                action = np.argmax(action)
                actions[action] = actions.get(action, 0) + 1

                current_state, reward, done, _ = self.env.step(action)
                current_state = self.convert_gray(current_state)
                self.env.render()
                score += reward

    def run(self):
        args = self.parser()
        self.initialize_env(args)
        if args.play:
            self.play()
        else:
            model = self.learn()
            self.save_model(model)


if __name__ == '__main__':
    agent = Agent()
    agent.run()
