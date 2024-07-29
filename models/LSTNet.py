import torch
import torch.nn as nn

class LSTNet(nn.Module):
    def __init__(self, input_size, seq_length, pred_length, skip_steps=2, ar_window=24, cnn_filters=100, cnn_kernel=6, lstm_units=100, skip_rnn_units=50, output_fun='Linear'):
        super(LSTNet, self).__init__()
        self.input_size = input_size
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.skip_steps = skip_steps
        self.ar_window = ar_window

        self.conv1d = nn.Conv1d(input_size, cnn_filters, kernel_size=cnn_kernel)
        self.lstm = nn.LSTM(cnn_filters, lstm_units, batch_first=True)
        self.skip_rnn = nn.LSTM(input_size * skip_steps, skip_rnn_units, batch_first=True)
        self.ar_dense = nn.Linear(ar_window * input_size, 100)
        self.output_dense = nn.Linear(lstm_units + skip_rnn_units + 100, pred_length)

        if output_fun == 'Linear':
            self.output_fun = None
        elif output_fun == 'Sigmoid':
            self.output_fun = nn.Sigmoid()
        else:
            raise ValueError("output_fun must be 'Linear' or 'Sigmoid'")

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Debug print

        # CNN component
        cnn_out = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        print(f"CNN output shape: {cnn_out.shape}")  # Debug print
        
        # LSTM component
        _, (lstm_out, _) = self.lstm(cnn_out)
        lstm_out = lstm_out.squeeze(0)
        print(f"LSTM output shape: {lstm_out.shape}")  # Debug print

        # Skip-RNN component
        skip_rnn_input = x.unfold(dimension=1, size=self.skip_steps, step=self.skip_steps).reshape(x.size(0), -1, self.skip_steps * self.input_size)
        print(f"Skip-RNN input shape: {skip_rnn_input.shape}")  # Debug print
        _, (skip_rnn_out, _) = self.skip_rnn(skip_rnn_input)
        skip_rnn_out = skip_rnn_out.squeeze(0)
        print(f"Skip-RNN output shape: {skip_rnn_out.shape}")  # Debug print

        # Autoregressive component
        ar_input = x[:, -self.ar_window:, :].contiguous().view(x.size(0), -1)
        ar_out = self.ar_dense(ar_input)
        print(f"AR output shape: {ar_out.shape}")  # Debug print

        # Concatenate all components
        concat_out = torch.cat([lstm_out, skip_rnn_out, ar_out], dim=1)
        print(f"Concatenated output shape: {concat_out.shape}")  # Debug print

        # Output layer
        output = self.output_dense(concat_out)
        print(f"Final output shape: {output.shape}")  # Debug print

        if self.output_fun:
            output = self.output_fun(output)

        return output