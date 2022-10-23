import torch
from torch import nn


class CNN_LSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_layer, output_size, batch_frist=True, bidirectional=False):
        super(CNN_LSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool1d(3, 1, 1)
        )
        self.lstm = nn.LSTM(out_channels, hidden_size, num_layer, batch_frist, bidirectional)
        if bidirectional:
            self.b = 2
        else:
            self.b = 1
        self.linear = nn.Linear(hidden_size * self.b, output_size)
        self.num_layers = num_layer
        self.hidden_size = hidden_size

    def forward(self, x):
        # conv对seq_len进行处理(-1)，所以要换下维度
        batch_size, seq_len, input_size = x.shape
        # h_0 = torch.randn(self.b * self.num_layers, seq_len, self.hidden_size)
        # c_0 = torch.randn(self.b * self.num_layers, seq_len, self.hidden_size)

        x = x.permute(0, 2, 1)
        x = self.conv(x)    # (b, out_channels, seq_len(卷积处理，相当于w，h))

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = x[:, -1, :]

        return x

if __name__ == '__main__':
    net = CNN_LSTM(in_channels=47, out_channels=32, hidden_size=32, num_layer=2, output_size=1)
    x = torch.randn((4, 20, 47))
    y = net(x)
    print(y.shape)