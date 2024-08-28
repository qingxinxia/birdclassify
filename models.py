from __future__ import annotations

from torch import nn
import torchinfo


class TinyNet(nn.Module):

    # partially hardcoded
    
    def __init__(self, num_ch_in: int, num_ch_out: int):
        super(TinyNet, self).__init__()
        
        self.num_ch_in = num_ch_in
        self.num_ch_out = num_ch_out
        
        self.b0_conv = nn.Conv1d(self.num_ch_in, 64, 2, padding=0, bias=True)
        self.b0_relu = nn.ReLU()
        self.b0_pool = nn.AvgPool1d(2, stride=2)

        self.b1_conv = nn.Conv1d(64, 64, 2, padding=0, bias=True)
        self.b1_relu = nn.ReLU()
        self.b1_pool = nn.AvgPool1d(2, stride=2)

        self.b2_conv = nn.Conv1d(64, 32, 2, padding=0, bias=True)
        self.b2_relu = nn.ReLU()
        self.b2_pool = nn.AvgPool1d(2, stride=2)

        self.b3_conv = nn.Conv1d(32, 16, 2, padding=0, bias=True)
        self.b3_relu = nn.ReLU()
        self.b3_pool = nn.AvgPool1d(2, stride=2)

        self.b4_conv = nn.Conv1d(16, 8, 2, padding=0, bias=True)
        self.b4_relu = nn.ReLU()
        self.b4_pool = nn.AvgPool1d(2, stride=2)

        self.flatten = nn.Flatten()

        self.fc0 = nn.Linear(48, 32, bias=True)
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
