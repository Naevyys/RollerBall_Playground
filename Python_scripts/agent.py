import torch
import numpy as np
from cnn import CNN


class Agent:
    def __init__(self, num: int, group, group_specs, vision_decoder: CNN = None) -> None:  # TODO: change vision_decoder
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

        observations_formatted = torch.tensor(observations).permute(2, 0, 1)  # Convert to tensor, move channels to first dim
        predictions = self.vision_decoder(observations_formatted)
        # Make additional computations here in order to cover entire range required (e.g. [-1, 1]) by using the predictions between [0, 1]
        return predictions.detach().numpy()

    def update_encoder(self):
        # Do a gradient step to update the cnn
        # Might be better to do that from the trainer directly
        raise NotImplementedError
    
    def increment_agent_num(self, amount = 1):
        '''
        To keep up with the incrementation of agent ids after environment reset.
        '''
        assert self.num + amount >= 0, "Number of the agent should not be negative."
        self.num += amount