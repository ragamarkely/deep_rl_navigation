from collections import deque
from typing import NamedTuple
import numpy as np
import random

from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # parameter for soft update of QNetwork target
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often network will be updated

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """
    Double DQN agent that can interact with and learn from the environment.
    """

    def __init__(self, state_size: int, action_size: int, seed: int) -> None:
        """
        Initialize an Agent object.
        
        Params
        ======
        state_size: number of states
        action_size: number of actions
        seed: random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_prime = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step for updating parameters
        self.t_step = 0

    def step(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """
        Save experiences and learn.

        Params
        ======
        state: previous state
        action: action taken
        reward: reward as a result of the action
        next_state: next state as a result of the action
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # If enough samples are in available in memory, get random subset and learn.
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
            self.swap_qnetwork()

    def swap_qnetwork(self) -> None:
        """
        Randomly swap Q network with Q network prime so that each network 
        is used for action selection and policy evaluation, 
        each at ~50% of the episodes.
        """
        self.qnetwork, self.qnetwork_prime = self.qnetwork_prime, self.qnetwork

    def act(self, state: np.ndarray, eps: float = 0.0) -> int:
        """
        Return actions given the state based on current policy.

        Params
        ======
        state: current state
        eps: epsilon greedy parameter
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state).to(device)
        self.qnetwork.train()

        # Epsilon greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences: tuple, gamma: float) -> None:
        """
        Update Q Network parameters based on batch of experience tuples.
        
        Params
        ======
        experiences = tuple of (state, action, reward, next state, done)
        gamma: discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_prime(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork, self.qnetwork_prime, TAU)

    def soft_update(self, local_model: QNetwork, target_model: QNetwork, tau: float) -> None:
        """
        Soft update the Q network parameters/
        θ_target = τ * θ_local + (1 - τ) * θ_target

        Params
        ======
        local_model: model whose parameter values will be copied from
        target_model: model whose parameter values will be copied to
        tau: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Buffer to store experience tuples.
    """
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        """
        Initialize buffer.

        Params
        ======
        buffer_size: max size of buffer
        batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Add new experience to memory buffer.
        """
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self) -> None:
        """
        Randomly sample a batch of experiences from memory.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """
        Get size of memory buffer.
        """
        return len(self.memory)