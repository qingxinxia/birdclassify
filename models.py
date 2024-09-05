from __future__ import annotations

from torch import nn
import torchinfo
import torch.nn.functional as F
import torch
from torchvision import models, transforms
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (1, 45, 20) -> (1, 1, 45, 20)
        x = self.resnet(x)
        return x

class TinyNet(nn.Module):

    # partially hardcoded
    
    def __init__(self, num_ch_in: int, num_ch_out: int):
        super(TinyNet, self).__init__()
        
        self.num_ch_in = num_ch_in
        self.num_ch_out = num_ch_out
        
        self.b0_conv = nn.Conv1d(self.num_ch_in, 64, 3, padding=0, bias=True)
        self.b0_relu = nn.ReLU()
        self.b0_pool = nn.AvgPool1d(3, stride=2)

        self.b1_conv = nn.Conv1d(64, 64, 6, padding=0, bias=True)
        self.b1_relu = nn.ReLU()
        self.b1_pool = nn.AvgPool1d(2, stride=2)

        self.b2_conv = nn.Conv1d(64, 32, 12, padding=0, bias=True)
        self.b2_relu = nn.ReLU()
        self.b2_pool = nn.AvgPool1d(2, stride=2)

        self.b3_conv = nn.Conv1d(32, 16, 6, padding=0, bias=True)
        self.b3_relu = nn.ReLU()
        self.b3_pool = nn.AvgPool1d(2, stride=2)

        self.b4_conv = nn.Conv1d(16, 8, 2, padding=0, bias=True)
        self.b4_relu = nn.ReLU()
        self.b4_pool = nn.AvgPool1d(2, stride=2)

        self.flatten = nn.Flatten()

        self.fc0 = nn.Linear(24, 32, bias=True)
        self.fc0_relu = nn.ReLU()
        self.fc1 = nn.Linear(32, 32, bias=True)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(32, self.num_ch_out, bias=True)
        
    def forward(self, x):

        x = self.b0_pool(self.b0_relu(self.b0_conv(x)))
        x = self.b1_pool(self.b1_relu(self.b1_conv(x)))
        x = self.b2_pool(self.b2_relu(self.b2_conv(x)))
        x = self.b3_pool(self.b3_relu(self.b3_conv(x)))
        x = self.b4_pool(self.b4_relu(self.b4_conv(x)))

        x = self.flatten(x)
        x = self.fc0_relu(self.fc0(x))
        x = self.fc1_relu(self.fc1(x))
        y = self.fc2(x)
        
        return y


class SoundNet(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:
        Args:
            params: (Params) contains num_channels
        """
        super(SoundNet, self).__init__()
        self.num_channels = 32

        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        # self.conv1 = nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1)
        self.conv1 = Conv2dSame(1, self.num_channels, 6, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        # self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1, padding=1)
        self.conv2 = Conv2dSame(self.num_channels, self.num_channels * 2, 6, bias=False)
        self.bn2 = nn.BatchNorm2d(self.num_channels * 2)
        # self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1)

        outc = 30
        # self.conv3 = Conv2dSame(self.num_channels * 2, self.num_channels * 1, 3, bias=False)
        self.conv3 = Conv2dSame(self.num_channels * 2, outc, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(outc)
        self.conv4 = Conv2dSame(outc, 8, 3, bias=False)
        # self.conv4 = Conv2dSame(self.num_channels * 1, 8, 3, bias=False)
        self.bn4 = nn.BatchNorm2d(8)

        # self.conv5 = nn.Conv1d(8, 20, 21, bias=False)
        self.conv5 = nn.Conv1d(8, 20, 9, bias=False)
        self.maxpool = nn.MaxPool2d(8, stride=4)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(outc, 8, kernel_size=1, bias=False),
            # nn.Conv2d(self.num_channels * 1, 8, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, outc, kernel_size=1, bias=False),
            # nn.Conv2d(8, self.num_channels * 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # self.agpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, input):
        """
        This function defines how we use the components of our network to operate on an input batch.
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .
        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.
        Note: the dimensions after each step are provided
        """
        s = self.maxpool(input)
        #                                                  -> batch_size x 3 x 32 x 32
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = s.unsqueeze(1)  # batchsize, 224,224 to batchsize, 1,224,224
        s = self.bn1(self.conv1(s))  # batch_size x num_channels x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels x 16 x 16
        s = self.bn2(self.conv2(s))  # batch_size x num_channels*2 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*2 x 8 x 8
        s = self.bn3(self.conv3(s))  # batch_size x num_channels*4 x 8 x 8
        s = F.relu(F.max_pool2d(s, 1))  # batch_size x num_channels*4 x 4 x 4

        se = self.se(s)
        s = s * se

        s = self.bn4(self.conv4(s))  # batch_size x num_channels*4 x 8 x 8
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*4 x 4 x 4

        # flatten the output for each image
        s = s.view(s.shape[0], 8, -1)  # batch_size x 4*4*num_channels*4
        s = F.max_pool1d(s, 4)
        # print(s.size())
        # exit()
        # s = s.view(-1, 4 * 4 * self.num_channels * 4)  # batch_size x 4*4*num_channels*4

        s = self.conv5(s)

        # apply 2 fully connected layers with dropout
        # s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
        #               p=self.dropout_rate, training=self.training)  # batch_size x self.num_channels*4
        # s = self.fc2(s)  # batch_size x 10

        return torch.squeeze(s, 2)

class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:
        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = 32

        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        # self.conv1 = nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1)
        self.conv1 = Conv2dSame(1, self.num_channels, 6, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        # self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1, padding=1)
        self.conv2 = Conv2dSame(self.num_channels, 48, 6, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        # self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1)

        outc = 24
        # outc = self.num_channels * 1
        self.conv3 = Conv2dSame(48, outc, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(outc)
        self.conv4 = Conv2dSame(outc, 8, 3, bias=False)
        self.bn4 = nn.BatchNorm2d(8)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        # self.fc1 = nn.Linear(4 * 4 * self.num_channels * 4, self.num_channels * 4)
        # self.fc1 = nn.Linear(50176, self.num_channels * 4)
        # self.fc1 = nn.Linear(648, 8)
        # self.fcbn1 = nn.BatchNorm1d(8)
        # self.fc2 = nn.Linear(8, 20)
        # self.dropout_rate = 0.5

        self.conv5 = nn.Conv1d(8, 20, 24, bias=False)
        self.maxpool = nn.MaxPool2d(4, stride=2)

        self.se = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(outc, 8, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, outc, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # self.agpool = nn.AdaptiveAvgPool2d((1,1))


    def forward(self, input):
        """
        This function defines how we use the components of our network to operate on an input batch.
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .
        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.
        Note: the dimensions after each step are provided
        """
        s = self.maxpool(input)
        #                                                  -> batch_size x 3 x 32 x 32
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = s.unsqueeze(1)  # batchsize, 224,224 to batchsize, 1,224,224
        s = self.bn1(self.conv1(s))  # batch_size x num_channels x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels x 16 x 16
        s = self.bn2(self.conv2(s))  # batch_size x num_channels*2 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*2 x 8 x 8
        s = self.bn3(self.conv3(s))  # batch_size x num_channels*4 x 8 x 8
        # s = F.relu(F.max_pool2d(s, 1))  # batch_size x num_channels*4 x 4 x 4
        s =  F.relu(s)
        se = self.se(s)
        s = s*se

        s = self.bn4(self.conv4(s))  # batch_size x num_channels*4 x 8 x 8
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*4 x 4 x 4

        # flatten the output for each image
        s = s.view(s.shape[0], 8, -1)  # batch_size x 4*4*num_channels*4
        s = F.max_pool1d(s, 7)
        # s = s.view(-1, 4 * 4 * self.num_channels * 4)  # batch_size x 4*4*num_channels*4

        s = self.conv5(s)

        # apply 2 fully connected layers with dropout
        # s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
        #               p=self.dropout_rate, training=self.training)  # batch_size x self.num_channels*4
        # s = self.fc2(s)  # batch_size x 10

        return torch.squeeze(s, 2)

# class Net_xia(nn.Module):
#     """
#     This is the standard way to define your own network in PyTorch. You typically choose the components
#     (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
#     on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
#
#     such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
#     step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
#     you can go about defining your own network.
#
#     The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
#     """
#
#     def __init__(self):
#         """
#         We define an convolutional network that predicts the sign from an image. The components
#         required are:
#
#         Args:
#             params: (Params) contains num_channels
#         """
#         super(Net_xia, self).__init__()
#         self.num_channels = 16
#
#         # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
#         # stride, padding). We also include batch normalisation layers that help stabilise training.
#         # For more details on how to use these layers, check out the documentation.
#         # self.conv1 = nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1)
#         self.conv1 = nn.Conv2d(1, self.num_channels, 4,
#                                stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(self.num_channels)
#         # self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 4,
#                                stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(self.num_channels * 2)
#         # self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 2,
#                                2, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(self.num_channels * 2)
#
#         # 2 fully connected layers to transform the output of the convolution layers to the final output
#         # self.fc1 = nn.Linear(4 * 4 * self.num_channels * 4, self.num_channels * 4)
#         # self.fc1 = nn.Linear(50176, self.num_channels * 4)
#         self.fc1 = nn.Linear(3136, self.num_channels * 2)
#         self.fcbn1 = nn.BatchNorm1d(self.num_channels * 2)
#         self.fc2 = nn.Linear(self.num_channels * 2, 20)
#         self.dropout_rate = 0.5
#
#     def forward(self, s):
#         """
#         This function defines how we use the components of our network to operate on an input batch.
#
#         Args:
#             s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .
#
#         Returns:
#             out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.
#
#         Note: the dimensions after each step are provided
#         """
#         #                                                  -> batch_size x 3 x 32 x 32
#         # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
#         s = s.unsqueeze(1)  # batchsize, 224,224 to batchsize, 1,224,224
#         s = self.bn1(self.conv1(s))  # batch_size x num_channels x 32 x 32
#         s = F.dropout(F.relu(F.max_pool2d(s, 2)) ,
#                     p=0.3, training=self.training)  # batch_size x num_channels x 16 x 16
#         s1 = self.bn2(self.conv2(s))  # batch_size x num_channels*2 x 16 x 16
#         s1 = F.dropout(F.relu(F.max_pool2d(s1, 2)),
#                     p=0.3, training = self.training)# batch_size x num_channels*2 x 8 x 8
#
#
#         s2 = self.bn3(self.conv3(s1))  # batch_size x num_channels*4 x 8 x 8
#         s2 = F.dropout(F.relu(F.max_pool2d(s2, 2)),
#                     p=0.3, training = self.training)  # batch_size x num_channels*4 x 4 x 4
#
#
#
#         # flatten the output for each image
#         s3 = s2.view(s2.shape[0], -1)  # batch_size x 4*4*num_channels*4
#         # s = s.view(-1, 4 * 4 * self.num_channels * 4)  # batch_size x 4*4*num_channels*4
#
#         # apply 2 fully connected layers with dropout
#         result = F.dropout(F.relu(self.fcbn1(self.fc1(s3))),
#                       p=self.dropout_rate, training=self.training)  # batch_size x self.num_channels*4
#         result = self.fc2(result)  # batch_size x 10
#
#         return result
#
#
# class Net_xia0829(nn.Module):
#     """
#     This is the standard way to define your own network in PyTorch. You typically choose the components
#     (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
#     on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
#
#     such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
#     step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
#     you can go about defining your own network.
#
#     The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
#     """
#
#     def __init__(self):
#         """
#         We define an convolutional network that predicts the sign from an image. The components
#         required are:
#
#         Args:
#             params: (Params) contains num_channels
#         """
#         super(Net_xia0829, self).__init__()
#         self.num_channels = 32
#
#         # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
#         # stride, padding). We also include batch normalisation layers that help stabilise training.
#         # For more details on how to use these layers, check out the documentation.
#         # self.conv1 = nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1)
#         # self.conv1 = nn.Conv2d(1, self.num_channels, 6, stride=3, padding=1)
#         self.conv1 = Conv2dSame(1, self.num_channels, 12)
#         # self.bn1 = nn.LayerNorm(72)
#         self.bn1 = nn.BatchNorm2d(self.num_channels)
#         # self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1, padding=1)
#         # self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 12, stride=3, padding=1)
#         self.conv2 = Conv2dSame(self.num_channels, self.num_channels * 1, 6)
#         # self.bn2 = nn.LayerNorm(17)
#         self.bn2 = nn.BatchNorm2d(self.num_channels * 1)
#         # self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1)
#         # self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 2, 3, stride=2, padding=1)
#         self.conv3 = Conv2dSame(self.num_channels * 1, 16, 6)
#         # self.bn3 = nn.LayerNorm(4)
#         self.bn3 = nn.BatchNorm2d(16)
#
#         # 2 fully connected layers to transform the output of the convolution layers to the final output
#         # self.fc1 = nn.Linear(4 * 4 * self.num_channels * 4, self.num_channels * 4)
#         # self.fc1 = nn.Linear(50176, self.num_channels * 4)
#         self.fc1 = nn.Linear(256, 16)
#         self.fcbn1 = nn.BatchNorm1d(16)
#         self.fc2 = nn.Linear(16, 20)
#         self.dropout_rate = 0.5
#         self.prelu = nn.PReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#
#         # self.conv4 = nn.Conv2d(self.num_channels * 1, 20,
#         #                        3)
#         # self.bn4 = nn.BatchNorm2d(20)
#         # self.conv5 = nn.Conv2d(self.num_channels, self.num_channels,
#         #                        (1,12), stride=3, padding=1)
#         # self.conv6 = nn.Conv2d(self.num_channels, self.num_channels,
#         #                        (1,1), stride=1, padding=1)
#         # self.conv7 = nn.Conv2d(self.num_channels * 2, self.num_channels * 2,
#         #                        (1, 1), stride=1, padding=1)
#         self.maxpool = nn.MaxPool2d(4, stride=2)
#
#     def forward(self, input):
#         """
#         This function defines how we use the components of our network to operate on an input batch.
#         Args:
#             s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .
#         Returns:
#             out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.
#         Note: the dimensions after each step are provided
#         """
#         #                                                  -> batch_size x 3 x 32 x 32
#         input = self.maxpool(input)
#         # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
#         s = input.unsqueeze(1)  # batchsize, 224,224 to batchsize, 1,224,224
#         s = self.bn1(self.conv1(s))  # batch_size x num_channels x 32 x 32
#         # s = F.dropout(F.relu(F.max_pool2d(s, 2)), p=0.2)  # batch_size x num_channels x 16 x 16
#         # s = F.dropout(self.prelu(F.max_pool2d(s, 2)), p=0.2)  # batch_size x num_channels x 16 x 16
#         s = F.dropout(self.sigmoid(F.max_pool2d(s, 4)), p=0.2)  # batch_size x num_channels x 16 x 16
#         # s = self.prelu(self.bn1(self.conv6(s)))
#
#         s = self.bn2(self.conv2(s))  # batch_size x num_channels*2 x 16 x 16
#         # s = F.dropout(F.relu(F.max_pool2d(s, 2)), p=0.2)  # batch_size x num_channels*2 x 8 x 8
#         # s = F.dropout(self.prelu(F.max_pool2d(s, 2)), p=0.2)  # batch_size x num_channels*2 x 8 x 8
#         s = F.dropout(self.sigmoid(F.max_pool2d(s, 3)), p=0.2)  # batch_size x num_channels*2 x 8 x 8
#         # s = self.prelu(self.bn2(self.conv7(s)))
#
#         s = self.bn3(self.conv3(s))  # batch_size x num_channels*4 x 8 x 8
#         # s = F.dropout(F.relu(F.max_pool2d(s, 2)), p=0.2)  # batch_size x num_channels*4 x 4 x 4
#         s = F.dropout(self.sigmoid(F.max_pool2d(s, 2)), p=0.2)  # batch_size x num_channels*4 x 4 x 4
#
#         # a = input.unsqueeze(1)  # batchsize, 224,224 to batchsize, 1,224,224
#         # aa = self.conv4(a)  # batch_size x num_channels x 32 x 32
#         # b = self.conv5(aa)  # batch_size x num_channels x 32 x 32
#
#         # s = self.bn4(self.conv4(s))
#         # s = F.dropout(self.prelu(F.max_pool2d(s, 2)), p=0.2)
#         # flatten the output for each image
#         s = s.view(s.shape[0], -1)  # batch_size x 4*4*num_channels*4
#         # s = F.max_pool1d(s, 2)
#
#         # apply 2 fully connected layers with dropout
#         # s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
#         s = F.dropout(self.prelu(self.fcbn1(self.fc1(s))),
#                       p=self.dropout_rate, training=self.training)  # batch_size x self.num_channels*4
#         s = self.fc2(s)  # batch_size x 10
#
#         return s

class ConvFuseNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.channel_num = 8

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.channel_num, (6, 1), (3, 1)),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.channel_num),
            nn.MaxPool2d(2),

            nn.Conv2d(self.channel_num, self.channel_num * 2, (3, 1), (2, 1)),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.channel_num * 2),
            # nn.MaxPool2d((2, 1), stride=(2, 1)),

            nn.Conv2d(self.channel_num * 2, self.channel_num * 4, (3, 1), (2, 1)),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.channel_num * 4),
            # nn.MaxPool2d(2)
            nn.ReLU(inplace=True),  # ))#,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, self.channel_num, (1, 6),(1,3)),
            # nn.ReLU(inplace=True),#))#,
            nn.BatchNorm2d(self.channel_num),
            nn.MaxPool2d(2),

            nn.Conv2d(self.channel_num, self.channel_num*2, (1, 3),(1,2)),
            # nn.ReLU(inplace=True),#))#,
            nn.BatchNorm2d(self.channel_num*2),
            # nn.MaxPool2d(2),

            nn.Conv2d(self.channel_num*2, self.channel_num*4, (1, 3),(1,2)),
            # nn.ReLU(inplace=True),#))#,
            nn.BatchNorm2d(self.channel_num*4),
            # nn.MaxPool2d(2)
            # nn.MaxPool2d(2)
            nn.ReLU(inplace=True),  # ))#,
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(1, self.channel_num, 6,3),
            # nn.ReLU(inplace=True),#))#,
            nn.BatchNorm2d(self.channel_num),
            nn.MaxPool2d(2),

            nn.Conv2d(self.channel_num, self.channel_num*2, 3,2),
            # nn.ReLU(inplace=True),#))#,
            nn.BatchNorm2d(self.channel_num*2),
            # nn.MaxPool2d(2),

            nn.Conv2d(self.channel_num*2, self.channel_num*4, 3, 2),
            # nn.ReLU(inplace=True),#))#,
            nn.BatchNorm2d(self.channel_num*4),
            # nn.MaxPool2d(2)
            # nn.MaxPool2d(2)
            nn.ReLU(inplace=True),  # ))#,
        )



        self.fc = nn.Sequential(
            nn.Linear(480,20),
            # nn.ReLU(),
            # nn.Linear(32,20)
        )

        self.maxpool = nn.MaxPool2d(4, stride=4)

    def forward(self,x):

        x = x.unsqueeze(1)

        x = self.maxpool(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x1 = F.max_pool2d(x1, (1,4))
        x2 = F.max_pool2d(x2, (4,1))
        # x3 = F.max_pool2d(x3, (2,2))


        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
  
        x = torch.cat((x1,x2,x3),dim=1)
        
        # print(x.size())
        # exit()
        out = self.fc(x)

        return out



class Conv2dSame(torch.nn.Module):
    '''2D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    2D convolution is represented by a depthwise convolution followed by a 1x1 pointwise convolution,
     i.e. a depthwise separable convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''

        # replace standard convolution with depthwise and pointwise convolution
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            # depthwise convolution by changing amount of groups
            nn.Conv2d(in_channels, in_channels, kernel_size, bias=bias, stride=1, groups=in_channels),
            # 1x1 convolution of input channels to output channels
            nn.Conv2d(in_channels, out_channels, 1)
        )

        self.weight = self.net[1].weight
        self.bias = self.net[1].bias

    def forward(self, x):
        return self.net(x)


def summarize(
    model: nn.Module,
    input_size: tuple[int],
    verbose: 0 | 1 | 2 = 0,
) -> torchinfo.ModelStatistics:

    # set all parameters for torchsummary

    batch_dim = 0  # index of the batch dimension
    col_names = [
        'input_size',
        'output_size',
        'num_params',
        'params_percent',
        'kernel_size',
        'mult_adds',
        'trainable',
    ]
    device = 'cpu'
    mode = 'eval'
    row_settings = [
        'ascii_only',
        'depth',
        'var_names',
    ]

    # call the summary function

    model_stats = torchinfo.summary(
        model=model,
        input_size=input_size,
        batch_dim=batch_dim,
        col_names=col_names,
        device=device,
        mode=mode,
        row_settings=row_settings,
        verbose=verbose,
    )

    return model_stats


def main() -> None:

    num_ch_in = 224
    num_samples_in = 224
    num_ch_out = 20

    # tinynet = TinyNet(num_ch_in, num_ch_out)
    tinynet = ConvFuseNet()
    tinynet.eval()

    input_size = (num_ch_in, num_samples_in)
    verbose = 2
    summarize(tinynet, input_size=input_size, verbose=verbose)


if __name__ == "__main__":
    main()
