from typing import Tuple
import torch
from torch.nn import Conv2d, ReLU, Linear, Module
from torch.nn.functional import softmax
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
        out = torch.zeros((1, self.output_size))
        out[0, self.tanh_pos] = tanh(input[0, self.tanh_pos])
        out[0, self.sigmoid_pos] = sigmoid(input[0, self.sigmoid_pos])
        out[0, self.relu_pos] = relu(input[0, self.relu_pos])
        out[0, self.softmax_pos] = softmax(input[0, self.softmax_pos], dim=0)
        return out


class CNN(Module):
    # Note: For now I'm almost exclusively reusing the code from the colab notebook, but I can always change the cnn later
    def __init__(self, input_shape: Tuple[int, int, int], encoding_size: int, output_size: int,
                tanh_positions: list = [1], sigmoid_positions: list = [0], relu_positions: list = []) -> None:
        # By default, I have position 0 of output in [0, 1] for movement and position 1 of output in [-1, 1] for rotation
        assert len(tanh_positions + sigmoid_positions + relu_positions) <= output_size, "Number of positions for final activation layer is greater than output size."
        super().__init__()
        height = input_shape[0]  # I have data structured as (height, width, channels)
        width = input_shape[1]
        initial_channels = input_shape[2]
        conv_1_hw = self.conv_output_shape((height, width), 8, 4)
        conv_2_hw = self.conv_output_shape(conv_1_hw, 4, 2)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32
        self.conv1 = Conv2d(initial_channels, 16, [8, 8], [4, 4])
        self.conv2 = Conv2d(16, 32, [4, 4], [2, 2])
        self.dense1 = Linear(self.final_flat, encoding_size)
        self.dense2 = Linear(encoding_size, output_size)
        self.final_activation = FinalActivation(output_size, tanh_positions, sigmoid_positions, relu_positions)
    
    def forward(self, observation):
        conv_1 = ReLU()(self.conv1(observation))
        conv_2 = ReLU()(self.conv2(conv_1))
        hidden = self.dense1(conv_2.reshape([-1, self.final_flat]))
        hidden = ReLU()(hidden)
        hidden = self.dense2(hidden)
        return self.final_activation(hidden)
    
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