import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # x.shape == [batch_size, in_channels, number of grid points]
        # hint: use torch.fft library torch.fft.rfft
        # use DFT to approximate the fourier transform

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        super(FNO1d, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 1  # pad the domain if input is non-periodic
        self.linear_p = nn.Linear(
            2, self.width
        )  # input channel is 2: (u0(x), x) --> GRID IS INCLUDED!

        self.spectral_layers = nn.ModuleList(
            [SpectralConv1d(self.width, self.width, self.modes1) for _ in range(4)]
        )

        self.linear_conv_layers = nn.ModuleList(
            [nn.Conv1d(self.width, self.width, 1) for _ in range(4)]
        )

        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, 1)

        self.activation = torch.nn.Tanh()

    def fourier_layer(self, x):
        ##########################################
        # TODO: Implement the Fourier layer:
        ##########################################
        for f, c in zip(self.spectral_layers, self.linear_conv_layers):
            x = f(x) + c(x)
            x = self.activation(x)

        return x

    def forward(self, x):
        #################################################
        # TODO: Implement the forward method
        #        using the Fourier and the Linear layer:
        #################################################

        # swap the channel dimension to the last dimension

        x = self.linear_p(x)

        x = x.permute(0, 2, 1)
        x = self.fourier_layer(x)
        x = x.permute(0, 2, 1)

        x = self.linear_q(x)

        return self.output_layer(x)
