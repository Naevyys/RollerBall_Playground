import torch
import numpy as np
from models import ActorCritic


class Agent:
    def __init__(self, num: int, group, group_specs, vision_decoder: ActorCritic = None) -> None:
        '''
        Initialize the agent with identifiers and its vision decoder.
        '''
        self.index = num  # This one will not be incremented
        self.num = num
        self.group = group
        self.group_specs = group_specs
        self.vision_decoder = vision_decoder

        if self.vision_decoder is None:
            print("No visual decoder specified, random actions will be sent instead.")
    
    def get_action(self, observations, action_size=2):
        '''
        Compute the actions from the observations using the vision decoder of the agent.
        '''
        if self.vision_decoder is None:
            return np.random.rand(action_size).reshape((1,-1))

        observations_formatted = torch.tensor(observations).permute(2, 0, 1).float()  # Convert to tensor, move channels to first dim
        means_vec, vars_vec, _ = self.vision_decoder(observations_formatted)
        actions = torch.normal(means_vec.detach(), torch.sqrt(vars_vec.detach()))  # Sample actions using means and stds
        return actions.numpy()
    
    def increment_agent_num(self, amount = 1):
        '''
        To keep up with the incrementation of agent ids after environment reset.
        '''
        assert self.num + amount >= 0, "Number of the agent should not be negative."
        self.num += amount