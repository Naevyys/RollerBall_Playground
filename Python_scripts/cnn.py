from typing import Tuple
from torch.nn import Conv2d, ReLU, Linear, Module, Sigmoid
from math import floor


# Use softmax or sigmoid? Softmax might be more appropriate if we want the agent to either turn or move forward, while sigmoid is better if we want the agent to do both
# Check range of predictions needed to turn in both directions & move only forward
#   Note: Moving only forward is currently enforced in RollerAgent.cs, but it would be cool to enforce it in the predictions of the CNN directly instead, then we can also
#   change that later in case we want the agent to be able to move backwards too
#   For now, I will be using sigmoid


class CNN(Module):
    # Note: For now I'm almost exclusively reusing the code from the colab notebook, but I can always change the cnn later
    def __init__(self, input_shape: Tuple[int, int, int], encoding_size: int, output_size: int) -> None:
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
    
    def forward(self, observation):
        conv_1 = ReLU()(self.conv1(observation))
        conv_2 = ReLU()(self.conv2(conv_1))  # Fails here (or step after it's not clear)
        hidden = self.dense1(conv_2.reshape([-1, self.final_flat]))
        hidden = ReLU()(hidden)
        hidden = self.dense2(hidden)
        return Sigmoid()(hidden)  # Just added a sigmoid here
    
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