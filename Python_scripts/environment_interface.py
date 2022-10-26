from mlagents_envs.environment import UnityEnvironment
from agent import Agent
from models import ActorCritic, ActorCriticLoss
from trainer import Trainer
from experience import Buffer
from typing import List
from torch.optim import Adam
import numpy as np


class EnvironmentInterface:
    def __init__(self, env_file_path: str) -> None:
        # Initialize the environment
        self.env_path = env_file_path
        self.terminal = False

    def load_env(self, side_channels: list = [], base_port=5004):
        self.env = UnityEnvironment(file_name=self.env_path, base_port=base_port, side_channels=side_channels)
        self.env.reset()

        # TODO: Get all observation and action shape information directly from the environment
        visual_field_shape = (20, 32, 1)
        actions_size = 2
        encoding_size = 64  # Arbitrary value, might need tuning here

        # Initialize agents
        self.agent_groups = []  # Each group has a single network/brain. Therefore, I consider each group as a single agent
        for group in self.env.get_agent_groups():
            vision_decoder = ActorCritic(visual_field_shape, encoding_size, actions_size)
            specs = self.env.get_agent_group_spec(group)
            self.agent_groups.append(Agent(0, group, specs, vision_decoder))

    def close_env(self):
        self.env.close()

    def reset_environment(self):
        '''
        Reset env after terminal state reached.
        '''
        self.env.reset()
        for agent in self.agent_groups:  # Increment all agents
            agent.increment_agent_num()  # If several agent groups, might need to increment by another amount than the default 1. There is an optional argument to change the amount

    def loop(self):
        '''
        - Get observations and corresponding actions for each agent
        - Step the environment
        - Check for terminal state, reset env if terminal state reached
        '''
        for agent in self.agent_groups:  # I should only have one
            batch_result = self.env.get_step_result(agent.group)
            step_result = batch_result.get_agent_step_result(agent.num)  # When agent gets reset, its ID changes...
            actions = agent.get_action(step_result.obs[0])
            self.env.set_actions(agent.group, actions)
        
        # Take step in environment
        self.env.step()

        for agent in self.agent_groups:
            # Collect result of step (rewards etc...)
            next_step_result = self.env.get_step_result(agent.group)
            reward = next_step_result.reward
            
            if not self.terminal:  # If False, set to terminal state of current agent. If already True, no need for change.
                self.terminal = next_step_result.done[agent.index]
        if self.terminal:  # If any of the agent groups is done, reset the env
            self.reset_environment()
            self.terminal = False
    
    def get_agents(self):
        '''
        Returns the agent groups list
        '''
        return self.agent_groups

    def train(self, agent_index: int = 0, n_training_steps: int=10, lr: float=0.001, n_epochs: int=3, batch_size: int=32, gamma: float=0.9, n_new_exp: int=int(1e2), buffer_size:int=int(1e3), epsilon=0.1):  # TODO: increase default values for n_new_exp and buffer_size, I set them low now for testing
        agent = self.agent_groups[agent_index]  # Train a specific agent group, by default the first one.
        optimizer = Adam(agent.vision_decoder.parameters(), lr=lr)  # Initialize optimizer
        criterion = ActorCriticLoss()
        buffer: Buffer() = []  # Initialize buffer of experiments
        cumulative_rewards: List[float] = []  # Initialize list of cumulative rewards

        for n in range(n_training_steps):  # Loop for n_training_steps
            new_exp, _ = Trainer.generate_trajectories(self.env, agent, n_new_exp, epsilon)  # Generate new experiments using trainer
            if len(buffer) > (buffer_size - n_new_exp):
                keep = np.random.choice(len(buffer), buffer_size - n_new_exp)  # Select (buffer_size - n_new_exp) experiments from buffer to keep
                buffer = buffer[keep]
            buffer.extend(new_exp)  # Add new ones to buffer
            Trainer.update_network(agent.vision_decoder, optimizer, buffer, n_epochs, batch_size, gamma, criterion)  # Update network using trainer
            _, mean_reward = Trainer.generate_trajectories(self.env, agent, 100, 0)  # Run new experiments and compute mean cumulative reward
            cumulative_rewards.append(mean_reward)  # Append cumulative reward to list of cumulative rewards
            print("Training step ", n+1, "\treward ", mean_reward)  # Print some feedback

    def run_steps(self, num_steps = 10):
        assert self.env is not None, "Env is not loaded, call load_env() first."
        # Just for testing, will be removed afterwards
        for _ in range(num_steps):
            self.loop()
