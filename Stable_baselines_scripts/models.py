from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch
from torch.nn import Conv2d, ReLU, Linear, Module, Sequential, Sigmoid, Tanh
from torch.nn.functional import softmax
from torch import tanh, sigmoid, relu
from math import floor
import numpy as np
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym


class CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int=64):
        super().__init__(observation_space, features_dim)
        initial_channels, height, width = observation_space.shape  # Pre-processing re-orders data as CxHxW

        # Initialize CNN
        conv_1_hw = self.conv_output_shape((height, width), 8, 4)
        conv_2_hw = self.conv_output_shape(conv_1_hw, 4, 2)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32
        self.conv1 = Sequential(Conv2d(initial_channels, 16, [8, 8], [4, 4]), ReLU())
        self.conv2 = Sequential(Conv2d(16, 32, [4, 4], [2, 2]), ReLU())
        self.encoding = Sequential(Linear(self.final_flat, features_dim), ReLU())
    
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


# UNUSED

class FinalActivation(Module):  # Doesn't work to bound the action values this way cause action choice is stochastic and follows a normal distribution that overflow the ranges
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


class ActorCritic(Module):
    def __init__(self, encoding_size: int, actions_size: int,
                tanh_positions: list = [1], sigmoid_positions: list = [0], relu_positions: list = []) -> None:
        # By default, I have position 0 of output in [0, 1] for movement and position 1 of output in [-1, 1] for rotation
        assert len(tanh_positions + sigmoid_positions + relu_positions) <= actions_size, "Number of positions for final activation layer is greater than output size."

        super().__init__()

        self.features_dim = encoding_size
        self.latent_dim_pi = actions_size
        self.latent_dim_vf = 1
        intermediate = int(encoding_size/2)

        self.actor = Sequential(
            Linear(self.features_dim, intermediate), 
            Sigmoid(), 
            Linear(intermediate, self.latent_dim_pi), 
            FinalActivation(actions_size, tanh_positions, sigmoid_positions, relu_positions)
            )  # Action must be in the range constraints for the different actions
        self.critic = Sequential(
            Linear(self.features_dim, intermediate),
            Sigmoid(),
            Linear(intermediate, self.latent_dim_vf)
            )  # Estimated TD-error computed by the critic

    def forward(self, observation):
        return self.actor(observation), self.critic(observation)
    
    def forward_actor(self, observation):
        return self.actor(observation)
    
    def forward_critic(self, observation):
        return self.critic(observation)


class CustomActorCriticPolicy(ActorCriticCnnPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[Module] = Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ActorCritic(self.features_dim, self.action_space.shape[0])
