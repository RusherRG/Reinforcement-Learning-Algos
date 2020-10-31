import argparse
import argcomplete
import gym
import numpy as np


class QLearning():
    def __init__(self):
        self.episodes = None
        self.env = None
        self.learning_rate = None
        self.discount = None
        self.results_every_n_episodes = 2000
        self.epsilon = 1

    def parser(self):
        argparser = argparse.ArgumentParser()
        argparser.add_argument(
            '-e',
            '--env',
            type=str,
            default='CartPool-v0',
            required=True,
            help='OpenAI Gym environment'
        )
        argparser.add_argument(
            '-ep',
            '--episodes',
            type=int,
            default=10000,
            help='Number of episodes to run'
        )
        argparser.add_argument(
            '-lr',
            '--learning_rate',
            type=float,
            default=0.1,
            help='Learning rate'
        )
        argparser.add_argument(
            '-d',
            '--discount',
            type=float,
            default=0.95,
            help='Discount'
        )
        argparser.add_argument(
            '--size',
            type=int,
            default=50,
            help='Size of state space'
        )
        argcomplete.autocomplete(argparser)
        return argparser.parse_args()

    def initialize_env(self, args):
        self.env = args.env
        self.episodes = args.episodes
        self.learning_rate = args.learning_rate
        self.discount = args.discount
        self.epsilon_decay_value = self.epsilon/(self.episodes//2)

        self.env = gym.make(self.env)
        discrete_state_space_size = [50] * len(self.env.observation_space.high)
        self.window = (self.env.observation_space.high -
                       self.env.observation_space.low)/discrete_state_space_size
        q_table_size = discrete_state_space_size + [self.env.action_space.n]

        # rewards = []
        # for _ in range(100):
        #     done = False
        #     env.reset()
        #     while not done:
        #         action = env.action_space.sample()
        #         _, reward, done, _ = env.step(action)
        #         rewards.append(reward)

        # q_table = np.random.uniform(low=min(rewards), high=max(rewards), size=q_table_size)
        self.q_table = np.random.uniform(low=0, high=1, size=q_table_size)

        return

    def get_discrete_state(self, state):
        discrete_state = (state - self.env.observation_space.low)/self.window
        return tuple(map(int, discrete_state))

    def learn(self):
        stats = {'scores': [], 'avg': [], 'min': [], 'max': []}
        for ep in range(self.episodes):
            state_t = self.get_discrete_state(self.env.reset())
            done = False
            score = 0
            while not done:

                if np.random.random() >  self.epsilon:
                    action = np.argmax(self.q_table[state_t])
                else:
                    action = self.env.action_space.sample()
                # action = np.argmax(self.q_table[state_t])
                current_q_value = np.max(self.q_table[state_t])

                state, reward, done, _ = self.env.step(action)
                if ep % self.results_every_n_episodes == 0:
                    self.env.render()
                score += reward

                state_t1 = self.get_discrete_state(state)
                max_q_value = np.max(self.q_table[state_t1])

                new_q_value = (1 - self.learning_rate)*current_q_value + \
                    self.learning_rate*(reward + self.discount*max_q_value)
                self.q_table[state_t + (action, )] = new_q_value

                state_t = state_t1

            if ep < self.episodes//2:
                self.epsilon -= self.epsilon_decay_value

            stats['scores'].append(score)
            if ep % self.results_every_n_episodes == 0:
                stats['avg'].append(sum(
                    stats['scores'][-self.results_every_n_episodes:])/self.results_every_n_episodes)
                stats['min'].append(
                    min(stats['scores'][-self.results_every_n_episodes:]))
                stats['max'].append(
                    max(stats['scores'][-self.results_every_n_episodes:]))
                print("Episode: {}\tAverage: {}\tMin: {}\tMax: {}".format(
                    ep, stats['avg'][-1], stats['min'][-1], stats['max'][-1]))
            # else:
            #     print("Episode: {}".format(ep))
        self.env.close()
        
        return

    def play(self):
        return

    def run(self):
        args = self.parser()
        self.initialize_env(args)
        self.learn()


if __name__ == '__main__':
    qlearning = QLearning()
    qlearning.run()
