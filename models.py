from __future__ import annotations

from torch import nn
import torchinfo
import torch.nn.functional as F


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
        self.num_channels = 16

        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        # self.conv1 = nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(1, self.num_channels, 12, stride=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        # self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 6, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels * 2)
        # self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 4, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels * 4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        # self.fc1 = nn.Linear(4 * 4 * self.num_channels * 4, self.num_channels * 4)
        # self.fc1 = nn.Linear(50176, self.num_channels * 4)
        self.fc1 = nn.Linear(256, self.num_channels * 4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels * 4)
        self.fc2 = nn.Linear(self.num_channels * 4, 20)
        self.dropout_rate = 0.5

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 32 x 32
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = s.unsqueeze(1)  # batchsize, 224,224 to batchsize, 1,224,224
        s = self.bn1(self.conv1(s))  # batch_size x num_channels x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels x 16 x 16
        s = self.bn2(self.conv2(s))  # batch_size x num_channels*2 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*2 x 8 x 8
        s = self.bn3(self.conv3(s))  # batch_size x num_channels*4 x 8 x 8
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*4 x 4 x 4

        # flatten the output for each image
        s = s.view(s.shape[0], -1)  # batch_size x 4*4*num_channels*4
        # s = s.view(-1, 4 * 4 * self.num_channels * 4)  # batch_size x 4*4*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
                      p=self.dropout_rate, training=self.training)  # batch_size x self.num_channels*4
        s = self.fc2(s)  # batch_size x 10

        return s


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
    num_samples_in = 64
    num_ch_out = 20

    tinynet = TinyNet(num_ch_in, num_ch_out)
    tinynet.eval()

    input_size = (num_ch_in, num_samples_in)
    verbose = 2
    summarize(tinynet, input_size=input_size, verbose=verbose)


# if __name__ == "__main__":
#     main()
