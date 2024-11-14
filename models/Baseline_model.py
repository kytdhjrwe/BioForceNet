import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

pre_len = 1

# Define attention layer
class AttentionBlock(nn.Module):
    def __init__(self, input_dim, single_attention_vector=False):
        super(AttentionBlock, self).__init__()
        self.single_attention_vector = single_attention_vector
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        self.output_linear = nn.Linear(input_dim * 2, input_dim)

    def forward(self, x):
        a = self.attention(x)
        if self.single_attention_vector:
            a = a.mean(dim=1)
            a = a.repeat(1, x.size(1), 1)

        # Add linear transformation or activation function to process the connected output
        output = torch.cat([x, a], dim=-1)
        output = self.output_linear(output)
        return output

class BiLSTM_attention(nn.Module):
    def __init__(self, input_dims, time_steps, lstm_units):
        super(BiLSTM_attention, self).__init__()
        self.bilstm = nn.LSTM(input_dims, lstm_units, bidirectional=True, batch_first=True)
        self.attention_block = AttentionBlock(lstm_units * 2)
        self.fc = nn.Linear(lstm_units * 2 * time_steps, pre_len)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x, _ = self.bilstm(x)  # LSTM expects (batch_size, time_steps, input_dims)
        x = self.dropout(x)
        x = self.attention_block(x)  # Apply attention block
        x = x.contiguous().view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dims, lstm_units):
        super(CNN_BiLSTM, self).__init__()
        # The first convolutional layer
        self.conv1d_1 = nn.Conv1d(in_channels=input_dims, out_channels=64, kernel_size=3, padding=1)
        # The second convolutional layer
        self.conv1d_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # The third convolutional layer
        self.conv1d_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(0.1)

        self.bilstm = nn.LSTM(256, lstm_units, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(lstm_units * 2, pre_len)


    def forward(self, x):

        # The first convolutional layer
        x = self.conv1d_1(x.transpose(1, 2))  # Conv1d expects (batch_size, channels, time_steps)
        x = torch.relu(x)
        x = self.dropout(x)

        # The second convolutional layer
        x = self.conv1d_2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # The third convolutional layer
        x = self.conv1d_3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # LSTM layer
        x, _ = self.bilstm(x.transpose(1, 2))  # LSTM expects (batch_size, time_steps, input_dims)
        x = self.dropout(x)
        # Take the output of the last time step
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class CNN_LSTM(nn.Module):
    def __init__(self, input_dims, lstm_units):
        super(CNN_LSTM, self).__init__()
        # The first convolutional layer
        self.conv1d_1 = nn.Conv1d(in_channels=input_dims, out_channels=64, kernel_size=3, padding=1)
        # The second convolutional layer
        self.conv1d_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # The third convolutional layer
        self.conv1d_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(256, lstm_units, bidirectional=False, batch_first=True)

        self.fc = nn.Linear(lstm_units, pre_len)

    def forward(self, x):
        # The first convolutional layer
        x = self.conv1d_1(x.transpose(1, 2))  # Conv1d expects (batch_size, channels, time_steps)
        x = torch.relu(x)
        x = self.dropout(x)

        # The second convolutional layer
        x = self.conv1d_2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # The third convolutional layer
        x = self.conv1d_3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # LSTM layer
        x, _ = self.lstm(x.transpose(1, 2))  # LSTM expects (batch_size, time_steps, input_dims)
        x = self.dropout(x)

        # Take the output of the last time step
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class CNN(nn.Module):
    def __init__(self, input_dims):
        super(CNN, self).__init__()
        # The first convolutional layer
        self.conv1d_1 = nn.Conv1d(in_channels=input_dims, out_channels=64, kernel_size=3, padding=1)
        # The second convolutional layer
        self.conv1d_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # The third convolutional layer
        self.conv1d_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(0.1)
        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(256, pre_len)

    def forward(self, x):
        # The first convolutional layer
        x = self.conv1d_1(x.transpose(1, 2))  # Conv1d expects (batch_size, channels, time_steps)
        x = torch.relu(x)
        x = self.dropout(x)

        # The second convolutional layer
        x = self.conv1d_2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # The third convolutional layer
        x = self.conv1d_3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.global_avg_pool(x).squeeze(-1)  # 移除最后一个维度，形状变为 (batch_size, channels)

        # 全连接层
        x = self.fc(x)
        return x


if __name__ == '__main__':

    model = BiLSTM_attention(input_dims = 8, time_steps = 80, lstm_units = 64)
    print(model)
