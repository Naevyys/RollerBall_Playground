from tokenize import Double
from mlagents_envs.environment import UnityEnvironment
from agent import Agent
from cnn import CNN
import numpy as np

class Trainer:
    def __init__(self, env_file_path: str, side_channels: list = []) -> None:
        # Initialize the environment
        self.env_path = env_file_path
        self.env = UnityEnvironment(file_name=self.env_path, base_port=5004, side_channels=side_channels)
        self.env.reset()
        self.terminal = False

        # TODO: Get all observation and action shape information directly from the environment
        visual_field_shape = (20, 32, 1)
        actions_size = 2
        encoding_size = 64  # Arbitrary value, might need tuning here

        # Initialize agents
        self.agent_groups = []  # Each group has a single network/brain. Therefore, I consider each group as a single agent
        for group in self.env.get_agent_groups():
            vision_decoder = CNN(visual_field_shape, encoding_size, actions_size)
            specs = self.env.get_agent_group_spec(group)
            self.agent_groups.append(Agent(0, group, specs, vision_decoder))

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
            print(actions)
            self.env.set_actions(agent.group, actions)
            # TODO: Check what these groups are, cause from Tom's code, it looks like I have to pass agent.group here, does that mean we make predictions for an entire group at once?
            #       If so, I need to remove that second loop/motify some parts of my code
        
        # Take step in environment
        self.env.step()

        for agent in self.agent_groups:
            # Collect result of step (rewards etc...)
            next_step_result = self.env.get_step_result(agent.group)
            reward = next_step_result.reward
            #print(reward)  # To have some feedback for now
            
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
    
    def close_env(self):
        self.env.close()

    def run_steps(self, num_steps = 10):
        # Just for testing, will be removed afterwards
        for _ in range(num_steps):
            self.loop()
        self.close_env()
