from collections.abc import Sequence

import torch
from torch import Tensor, nn

##########################################################################################
################################# Configuration ##########################################
##########################################################################################


class AutoEncoderConfig:
    model_type = "autoencoder"

    def __init__(
        self,
        input_dim: int = 299,
        input_channel: int = 3,
        latent_dim: int = 2048,
        dropout_rate: float = 0.1,
        num_layers: int = 5,
        conv_layers: Sequence[int] = (32, 64, 128, 256, 512),
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.input_channel = input_channel
        self.conv_layers = conv_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Calculate the output dimensions after the encoder
        self.output_height, self.output_width = self.calculate_output_dimensions(
            input_dim, len(conv_layers), kernel_size, stride, padding
        )
        # print(self.output_height, self.output_width )

    @staticmethod
    def calculate_output_dimensions(input_dim, num_layers, kernel_size, stride, padding):
        output_height = input_dim
        output_width = input_dim
        for _ in range(num_layers):
            output_height = (output_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
            output_width = (output_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        return output_height, output_width


##########################################################################################
##################################### Model ##############################################
##########################################################################################


class AutoEncoder(nn.Module):
    def __init__(self, config: AutoEncoderConfig, ckpt: str | None = None):
        super().__init__()
        self.config = config

        # ------- Encoder -------
        layers = []
        in_channels = config.input_channel
        for out_channels in config.conv_layers:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.padding,
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels

        # layers.append(AttnBlock(out_channels))  # uncomment
        self.encoder = nn.Sequential(*layers)

        self.flattened_size = (
            config.output_height * config.output_width * config.conv_layers[-1]
        )  # 512*8*8 in this case
        self.fc_enc = nn.Sequential(
            nn.Linear(self.flattened_size, config.latent_dim), nn.BatchNorm1d(config.latent_dim), nn.ReLU()
        )

        # ------ End of Encoder ------

        # -------  Decoder  -------

        self.fc_dec = nn.Sequential(
            nn.Linear(config.latent_dim, self.flattened_size), nn.BatchNorm1d(self.flattened_size), nn.ReLU()
        )

        decoder_layers = []
        reversed_conv_layers = list(reversed(config.conv_layers))
        for i in range(len(reversed_conv_layers) - 1):
            output_padding = 1 if i == 1 or i == 3 else 0

            decoder_layers.append(
                nn.ConvTranspose2d(
                    in_channels=reversed_conv_layers[i],
                    out_channels=reversed_conv_layers[i + 1],
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.padding,
                    output_padding=output_padding,
                )
            )
            decoder_layers.append(nn.ReLU())

        decoder_layers.append(
            nn.ConvTranspose2d(
                in_channels=reversed_conv_layers[-1],
                out_channels=config.input_channel,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
                output_padding=0,
            )
        )
        decoder_layers.append(nn.Tanh())  #  output values are between 0 and 1
        self.decoder = nn.Sequential(*decoder_layers)

        if ckpt is not None:
            self.load_state_dict(torch.load(ckpt)["state_dict"])
            self.eval()

    def encode(self, inputs_ids) -> Tensor:
        latent = self.encoder(inputs_ids)
        latent = self.fc_enc(latent.reshape(latent.size(0), -1))
        return latent

    def forward(self, input_ids: Tensor, position_ids: Tensor | None = None, labels: Tensor | None = None) -> Tensor:
        labels = labels if labels != None else input_ids  # noqa: E711 - Tensor operand; keep the upstream comparison.

        # Encoding
        # print('what is feeded to the encoder is of shape', input_ids.shape)
        encoded = self.encoder(input_ids)
        encoded = encoded.reshape(encoded.size(0), -1)  # Flatten the output of the last conv layer
        encoded = self.fc_enc(encoded)  # Map to latent space

        # Decoding
        decoded = self.fc_dec(encoded)  # Map from latent space
        decoded = decoded.view(
            decoded.size(0), self.config.conv_layers[-1], self.config.output_height, self.config.output_width
        )
        output_ids = self.decoder(decoded)

        return output_ids
