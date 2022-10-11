import numpy as np


# Implement CNN here or in a new file


class Agent:
    def __init__(self, num: int, group, group_specs, vision_decoder = None) -> None:  # TODO: change vision_decoder
        self.index = num  # This one will not be incremented
        self.num = num
        self.group = group
        self.group_specs = group_specs
        self.vision_decoder = vision_decoder
    
    def get_action(self, observations):
        # Pass observations through cnn
        # return actions
        return np.random.rand(2).reshape((1,-1))
    
    def update_encoder(self):
        # Do a gradient step to update the cnn
        raise NotImplementedError
    
    def increment_agent_num(self, amount = 1):
        assert self.num + amount >= 0, "Number of the agent should not be negative."
        self.num += amount