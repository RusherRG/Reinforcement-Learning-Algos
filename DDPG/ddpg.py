import torch
import torch.nn as nn
import torch.optim as optim
from models import Actor, Critic, Memory

device = torch.device('cuda')

class DDPGAgent:
    def __init__(self, env, replay_memory_size, learning_rate, batch_size, gamma, tau):
        self.env = env
        self.state_size = self.env.observation_space.shape
        self.num_actions = self.env.action_space.shape[0]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau

        # initialize memory
        self.memory = Memory(replay_memory_size, batch_size)

        # create models
        self.actor = Actor(self.num_actions).to(device)  # policy network
        self.critic = Critic(self.num_actions).to(device)  # q network

        # create target models
        self.actor_target = Actor(self.num_actions).to(device)  # target policy network
        self.critic_target = Critic(self.num_actions).to(device)  # target q network

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.learning_rate)

    def get_action(self, state):
        state = torch.cuda.FloatTensor(state).unsqueeze(0)
        action = self.actor.forward(state)
        action = action.detach().cpu().numpy()[0]
        return action

    def update(self):
        states, actions, rewards, new_states = self.memory.get_batch()

        states = torch.cuda.FloatTensor(states)
        actions = torch.cuda.FloatTensor(actions)
        rewards = torch.cuda.FloatTensor([rewards, rewards, rewards]).transpose(0, 1)
        new_states = torch.cuda.FloatTensor(new_states)

        # q network loss (critic loss)
        q_val = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(states)
        new_q = self.critic_target.forward(states, next_actions.detach())
        new_q = rewards + self.gamma*new_q
        # print(new_q)
        # print(q_val)
        criterion = nn.MSELoss()
        critic_loss = criterion(q_val, new_q)

        # critic update (gradient descent)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # policy loss (actor loss)
        policy_loss = - \
            self.critic.forward(states, self.actor.forward(states)).mean()

        # actor update (gradient ascent)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        return
