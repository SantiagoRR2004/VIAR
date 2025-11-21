import torch
import torch.nn as nn
import torch.nn.functional as F

# ================== Part 1: U-Net Architecture ==================


class DoubleConv(nn.Module):
    """Two consecutive convolution layers with BatchNorm and ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Task 1.1: Implement double convolution block
        # Hint: Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
        # Use kernel_size=3, padding=1 to maintain spatial dimensions
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Encoder block with double convolution and pooling"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Task 1.1: Initialize layers
        # You need: DoubleConv and MaxPool2d
        self.conv = DoubleConv(in_channels, out_channels)
        if in_channels != out_channels:
            self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # Task 1.1: Implement forward pass
        # Return both: features before pooling (for skip connection) and after pooling
        features = self.conv(x)
        if not hasattr(self, "pool"):
            return features, features
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, skip connection, and double convolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        skipType: "FlexibleSkipConnection",
        upsampling="transpose",
    ):
        super().__init__()
        self.upsampling = upsampling
        self.skipType = skipType

        if in_channels != out_channels:
            # Task 1.2: Initialize upsampling layer
            if upsampling == "transpose":
                self.up = nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=2, stride=2
                )
            else:  # bilinear
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                )

        # Task 1.2: Initialize double convolution
        self.conv = DoubleConv(skipType.out_channels, out_channels)

    def forward(self, x, skip_features):
        # Task 1.2: Implement forward pass
        # 1. Upsample x
        # 2. Handle dimension mismatch if necessary (crop or pad)
        # 3. Concatenate with skip_features
        # 4. Apply double convolution
        if hasattr(self, "up"):
            x = self.up(x)

        x = self.skipType(x, skip_features)

        return self.conv(x)


class UNet(nn.Module):
    """Complete U-Net architecture"""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,  # each pixel has a probability for each class
        features=[64, 128, 256, 512],
        skipMode: str = "concat",
    ):
        super().__init__()
        self.skipMode = skipMode

        # Task 1.3: Build encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        # : Create encoder blocks
        # Hint: First block takes in_channels, others take features[i-1]
        self.encoders.append(EncoderBlock(in_channels, features[0]))
        for i in range(1, len(features)):
            self.encoders.append(EncoderBlock(features[i - 1], features[i]))

        # Task 1.3: Bottleneck (deepest part)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Task 1.3: Build decoder path
        self.decoders = nn.ModuleList()
        self.skipps: list[FlexibleSkipConnection] = nn.ModuleList()

        # Create decoder blocks (in reverse order)
        self.skipps.append(
            FlexibleSkipConnection(features[-1], features[-1], mode=skipMode)
        )
        self.decoders.append(
            DecoderBlock(features[-1] * 2, features[-1], skipType=self.skipps[-1])
        )  # 1024 -> 512
        for i in range(len(features) - 1, 0, -1):
            self.skipps.append(
                FlexibleSkipConnection(features[i - 1], features[i - 1], mode=skipMode)
            )
            self.decoders.append(
                DecoderBlock(features[i], features[i - 1], skipType=self.skipps[-1])
            )

        # Task 1.4: Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Task 1.4: Connect everything together
        skip_connections = []

        # Encoder path
        # Process through encoders, save skip connections
        for encoder in self.encoders:
            features, x = encoder(x)
            if self.skipMode != "none":
                skip_connections.append(features)

        # Bottleneck
        # Process through bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for decoding

        # Decoder path
        # Process through decoders with skip connections
        if self.skipMode != "none":
            for idx, decoder in enumerate(self.decoders):
                x = decoder(x, skip_connections[idx])
        else:
            for decoder in self.decoders:
                x = decoder(x, torch.zeros(1))  # Dummy skip features

        # Final layer
        # Apply final convolution
        x = self.final_conv(x)

        return x  # Return final output


# ================== Part 2: Skip Connection Strategies ==================


class AttentionGate(nn.Module):
    """Attention gate for skip connections"""

    def __init__(self, gate_channels, skip_channels):
        super().__init__()
        # Task 2.3: Implement attention gate
        # You need: Conv2d layers for gating signal, skip features, and psi
        inter = gate_channels // 2

        # Conv2d for gating signal
        self.W_g = nn.Conv2d(gate_channels, inter, 1)

        # Conv2d for skip features
        self.W_x = nn.Conv2d(skip_channels, inter, 1)

        # Conv2d for final attention coefficients
        self.psi = nn.Conv2d(inter, 1, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        # Task 2.3: Implement attention mechanism

        # 1. Process gate and skip through respective convolutions
        g1, x1 = self.W_g(gate), self.W_x(skip)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[2:], mode="bilinear", align_corners=True
            )

        # 2. Add and apply ReLU
        attention = self.relu(g1 + x1)

        # 3. Apply psi convolution and sigmoid
        attention = torch.sigmoid(self.psi(attention))

        # 4. Multiply with skip features
        return skip * attention


class FlexibleSkipConnection(nn.Module):
    """Flexible skip connection with different strategies"""

    def __init__(self, decoder_channels, skip_channels, mode="concat"):
        super().__init__()
        self.mode = mode
        self.out_channels = decoder_channels

        if mode == "concat":
            # Task 2.1: Setup for concatenation
            # Output conv to handle concatenated channels

            # Concat features from decoder and skip (encoder)
            # Preserve all information, higher computational cost
            # Use a conv layer 3 x 3 to reduce channels back to decoder_channels
            self.conv = nn.Conv2d(
                decoder_channels + skip_channels, decoder_channels, 3, padding=1
            )

        elif mode == "add":
            # Task 2.2: Setup for addition
            # May need 1x1 conv to match channels; to cast skip to decoder channels
            # Element-wise addition, lower computational cost, may lose some information
            self.proj = (
                nn.Conv2d(skip_channels, decoder_channels, 1)
                if skip_channels != decoder_channels
                else nn.Identity()
            )

        elif mode == "attention":
            # Task 2.3: Setup attention gate
            self.attention_gate = AttentionGate(decoder_channels, skip_channels)
            self.conv = nn.Conv2d(
                decoder_channels + skip_channels, decoder_channels, 3, padding=1
            )

        elif mode == "none":
            # No skip connection - just pass through decoder features
            pass

    def forward(self, decoder_features, skip_features):
        # If mode is "none", skip all calculations and just return decoder features
        if self.mode == "none":
            return decoder_features

        # For other modes, we need to pad decoder features to match skip features
        diffY = skip_features.size()[2] - decoder_features.size()[2]
        diffX = skip_features.size()[3] - decoder_features.size()[3]

        decoder_features = F.pad(
            decoder_features,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

        # Implement forward pass based on mode
        if self.mode == "concat":
            return self.conv(torch.cat([decoder_features, skip_features], dim=1))

        elif self.mode == "add":
            return decoder_features + self.proj(skip_features)

        elif self.mode == "attention":
            return self.conv(
                torch.cat(
                    [
                        decoder_features,
                        self.attention_gate(decoder_features, skip_features),
                    ],
                    dim=1,
                )
            )
