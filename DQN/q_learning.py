import gym
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
import argparse
import argcomplete
import datetime

DROPOUT = 0.1
OUTPUT_LAYER = 18
REPLAY_MEMORY_SIZE = 25000
MINIBATCH_SIZE = 64
DISCOUNT = 0.95
UPDATE_TARGET_EVERY = 5
EPSILON = 1


class DQN:
    def __init__(self, env):
        self.env = env
        self.episode_counter = 0

        self.DROPOUT = DROPOUT
        self.OUTPUT_LAYER = OUTPUT_LAYER
        self.REPLAY_MEMORY_SIZE = REPLAY_MEMORY_SIZE
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.DISCOUNT = DISCOUNT
        self.UPDATE_TARGET_EVERY = UPDATE_TARGET_EVERY

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = []

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, (3, 3), activation='relu', input_shape=self.env.observation_space.shape),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(self.DROPOUT),

            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(self.DROPOUT),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(self.DROPOUT),

            tf.keras.layers.Dense(64, activation='relu'),

            tf.keras.layers.Dense(self.OUTPUT_LAYER, activation='linear')
        ])

        model.compile(optimizer='Adam',
                      loss='MSE', metrics=['accuracy'])

        return model

    # replay memeory
    def update_memory(self, memory):
        if len(self.replay_memory) > self.REPLAY_MEMORY_SIZE:
            self.replay_memory.pop()
        self.replay_memory.append(memory)
        return

    # train
    def train(self, episode_end):
        if len(self.replay_memory) < self.MINIBATCH_SIZE:
            return

        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        X, Y = [], []

        for (current_state, action, reward, done, new_state) in minibatch:

            if not done:
                # get the future q value
                future_q = self.target_model.predict(
                    np.array(new_state).reshape(-1, *new_state.shape)/255)[0]
                max_future_q = np.max(future_q)
                new_q = reward + self.DISCOUNT*max_future_q
            else:
                new_q = reward
            # get current q-value and update it with new q-value
            current_q = self.model.predict(
                np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
            current_q[action] = new_q

            X.append(current_state)
            Y.append(current_q)

        self.model.fit(np.array(X)/255, np.array(Y),
                       batch_size=self.MINIBATCH_SIZE)

        if episode_end:
            self.episode_counter += 1

        if self.episode_counter >= self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.episode_counter = 0


class Agent:
    def __init__(self):
        self.env = None
        self.episodes = None
        self.epsilon = EPSILON
        self.results_every_n_episodes = 10

    def parser(self):
        argparser = argparse.ArgumentParser()
        argparser.add_argument(
            '-e',
            '--env',
            type=str,
            default='Boxing-v0',
            help='OpenAI Gym environment'
        )
        argparser.add_argument(
            '-ep',
            '--episodes',
            type=int,
            default=500,
            help='Number of episodes to run'
        )
        argcomplete.autocomplete(argparser)
        args = argparser.parse_args()

        return args

    def initialize_env(self, args):
        self.env = gym.make(args.env)
        self.episodes = args.episodes
        self.epsilon_decay_value = 0.000001
    def learn(self):
        dqn = DQN(self.env)

        stats = {'scores': [], 'avg': [], 'min': [], 'max': []}
        for ep in tqdm(range(1, self.episodes + 1), ascii=True, unit='episodes'):
            print(self.epsilon)
            action_stats = [0, 0]
            current_state = self.env.reset()
            done = False
            score = 0
            steps = 0

            while not done:
                steps += 1
                current_q_values = dqn.model.predict(
                    np.array(current_state).reshape(-1, *current_state.shape)/255)[0]

                if np.random.random() > self.epsilon:
                    action_stats[0] += 1
                    action = np.argmax(current_q_values)
                else:
                    action_stats[1] += 1
                    action = self.env.action_space.sample()

                # print(action)
                new_state, reward, done, _ = self.env.step(action)
                if ep % self.results_every_n_episodes == 0:
                    self.env.render()
                # self.env.render()
                score += reward

                memory = (current_state, action, reward, done, new_state)
                dqn.update_memory(memory)

                if steps % 64 == 0:
                    dqn.train(done)

                current_state = new_state

                if self.epsilon > 0.1:
                    self.epsilon -= self.epsilon_decay_value

            print(action_stats)
            print(score)
            stats['scores'].append(score)
            # if ep % self.results_every_n_episodes == 0:
            #     stats['avg'].append(sum(
            #         stats['scores'][-self.results_every_n_episodes:])/self.results_every_n_episodes)
            #     stats['min'].append(
            #         min(stats['scores'][-self.results_every_n_episodes:]))
            #     stats['max'].append(
            #         max(stats['scores'][-self.results_every_n_episodes:]))
            #     print("Episode: {}\tAverage: {}\tMin: {}\tMax: {}".format(
            #         ep, stats['avg'][-1], stats['min'][-1], stats['max'][-1]))

        self.env.close()
        return dqn.model

    def save_model(self, model):
        model.save_weights(
            "./models/{}.model".format(str(datetime.datetime.now())))
        return

    def play(self, model):
        for _ in range(5):
            current_state = self.env.reset()
            done = False
            score = 0
            while not done:
                action = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
                print(action)
                action = np.argmax(action)
                print(action)
                current_state, reward, done, _ = self.env.step(action)
                self.env.render()

                score += reward
            print(score)
    
    def run(self):
        args = self.parser()
        self.initialize_env(args)
        model = self.learn()
        self.save_model(model)
        self.play(model)


if __name__ == '__main__':
    agent = Agent()
    agent.run()
