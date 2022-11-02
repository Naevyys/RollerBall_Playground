from typing import Tuple
import torch
from torch.nn import Conv2d, ReLU, Linear, Module, Sequential, Sigmoid, HuberLoss
from torch.nn.functional import softmax
from torch.distributions.normal import Normal
from torch import tanh, sigmoid, relu
from math import floor
import numpy as np


class FinalActivation(Module):
    def __init__(self, output_size: int, tanh_positions: list, sigmoid_positions: list, relu_positions: list) -> None:
        super().__init__()
        self.output_size = output_size
        self.tanh_pos = tanh_positions
        self.sigmoid_pos = sigmoid_positions
        self.relu_pos = relu_positions
        self.softmax_pos = list(np.delete(np.arange(self.output_size), self.tanh_pos + self.sigmoid_pos + relu_positions))
    
    def forward(self, input):
        '''
        Apply tanh on tanh_pos, sigmoid on sigmoid_pos, relu on relu_pos and a softmax on the remaining positions
        '''
        in_shape = input.shape
        out = torch.zeros((in_shape[0], self.output_size))
        out[:, self.tanh_pos] = tanh(input[:, self.tanh_pos])
        out[:, self.sigmoid_pos] = sigmoid(input[:, self.sigmoid_pos])
        out[:, self.relu_pos] = relu(input[:, self.relu_pos])
        out[:, self.softmax_pos] = softmax(input[:, self.softmax_pos], dim=0)
        return out


class CNN(Module):
    def __init__(self, input_shape: Tuple[int, int, int], encoding_size: int) -> None:
        super().__init__()
        height = input_shape[0]  # I have data structured as (height, width, channels)
        width = input_shape[1]
        initial_channels = input_shape[2]

        # Initialize CNN
        conv_1_hw = self.conv_output_shape((height, width), 8, 4)
        conv_2_hw = self.conv_output_shape(conv_1_hw, 4, 2)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32
        self.conv1 = Sequential(Conv2d(initial_channels, 16, [8, 8], [4, 4]), ReLU())
        self.conv2 = Sequential(Conv2d(16, 32, [4, 4], [2, 2]), ReLU())
        self.encoding = Sequential(Linear(self.final_flat, encoding_size), ReLU())
    
    def forward(self, observation):
        conv_1 = self.conv1(observation)
        conv_2 = self.conv2(conv_1)
        hidden = self.encoding(conv_2.reshape([-1, self.final_flat]))
        return hidden
    
    @staticmethod
    def conv_output_shape(
        h_w: Tuple[int, int],
        kernel_size: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
    ):
        """
        Computes the height and width of the output of a convolution layer.
        """
        h = floor(
        ((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        w = floor(
        ((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        return h, w


class ActorCritic(Module):
    def __init__(self, input_shape: Tuple[int, int, int], encoding_size: int, actions_size: int,
                tanh_positions: list = [1], sigmoid_positions: list = [0], relu_positions: list = []) -> None:
        # By default, I have position 0 of output in [0, 1] for movement and position 1 of output in [-1, 1] for rotation
        assert len(tanh_positions + sigmoid_positions + relu_positions) <= actions_size, "Number of positions for final activation layer is greater than output size."

        super().__init__()

        intermediate = int(encoding_size/2)

        self.encoder = CNN(input_shape, encoding_size)

        self.actor_mean = Sequential(
            Linear(encoding_size, intermediate), 
            Sigmoid(), 
            Linear(intermediate, actions_size), 
            FinalActivation(actions_size, tanh_positions, sigmoid_positions, relu_positions)
            )  # Mean for each action possible, must be in the range constraints for the different actions
        self.actor_var = Sequential(
            Linear(encoding_size, intermediate), 
            Sigmoid(),
            Linear(intermediate, actions_size), 
            Sigmoid()
            )  # Var for each action possible, can only be positive and I want to have it rather small since final actions should ideally remain in their mean range, therefore I use sigmoid to constraint it between 0 and 1.
        self.critic = Sequential(
            Linear(encoding_size, intermediate),
            Sigmoid(),
            Linear(intermediate, 1)
            )  # Estimated TD-error computed by the critic

    def forward(self, observation):
        encoding = self.encoder(observation)
        mean, var, td_err = self.actor_mean(encoding), self.actor_var(encoding), self.critic(encoding)
        #print("[INSIDE AcotrCritic forward] means: {}, vars: {}, td_error: {}".format(mean, var, td_err))
        return mean, var, td_err


class ActorCriticLoss(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, means, vars, actions, td_val, expected_td):
        # Following https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
        #print("[INSIDE ActorCriticLoss] means: {}, vars: {}".format(means, vars))
        advantage = expected_td - td_val
        loss_actor = torch.sum(Normal(means, torch.sqrt(vars)).log_prob(actions) * advantage)
        loss_critic = HuberLoss()(td_val, expected_td)  # Huber loss is less sensitive to outliers than plain MSE
        return loss_actor + loss_critic
