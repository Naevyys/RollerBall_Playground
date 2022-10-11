import numpy as np


# Implement CNN here or in a new file
# Use softmax or sigmoid? Softmax might be more appropriate if we want the agent to either turn or move forward, while sigmoid is better if we want the agent to do both
# Check range of predictions needed to turn in both directions & move only forward
#   Note: Moving only forward is currently enforced in RollerAgent.cs, but it would be cool to enforce it in the predictions of the CNN directly instead, then we can also
#   change that later in case we want the agent to be able to move backwards too


class Agent:
    def __init__(self, num: int, group, group_specs, vision_decoder = None) -> None:  # TODO: change vision_decoder
        '''
        Initialize the agent with identifiers and its vision decoder.
        '''
        self.index = num  # This one will not be incremented
        self.num = num
        self.group = group
        self.group_specs = group_specs
        self.vision_decoder = vision_decoder
    
    def get_action(self, observations, action_size=2):
        '''
        Compute the actions from the observations using the vision decoder of the agent.
        '''
        # Pass observations through cnn
        # return actions
        return np.random.rand(action_size).reshape((1,-1))
    
    def update_encoder(self):
        # Do a gradient step to update the cnn
        raise NotImplementedError
    
    def increment_agent_num(self, amount = 1):
        '''
        To keep up with the incrementation of agent ids after environment reset.
        '''
        assert self.num + amount >= 0, "Number of the agent should not be negative."
        self.num += amount