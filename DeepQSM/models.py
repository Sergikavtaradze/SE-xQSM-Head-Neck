"""
PyTorch implementation of QSM Deep Learning Models
Converted from TensorFlow/Keras implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleBlock(nn.Module):
    """Downsampling block with Conv3D + BatchNorm + LeakyReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, apply_batchnorm=True):
        super(DownsampleBlock, self).__init__()
        
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=2, 
            padding=kernel_size//2,
            bias=False
        )
        
        self.apply_batchnorm = apply_batchnorm
        if apply_batchnorm:
            self.batchnorm = nn.BatchNorm3d(out_channels)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x)
        x = self.leaky_relu(x)
        return x


class UpsampleBlock(nn.Module):
    """Upsampling block with Conv3DTranspose + BatchNorm + Dropout + ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, apply_dropout=False, apply_batchnorm=True):
        super(UpsampleBlock, self).__init__()
        
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size//2,
            output_padding=1,
            bias=False
        )
        
        self.apply_batchnorm = apply_batchnorm
        if apply_batchnorm:
            self.batchnorm = nn.BatchNorm3d(out_channels)
        
        self.apply_dropout = apply_dropout
        if apply_dropout:
            self.dropout = nn.Dropout3d(0.05)  # 5% dropout as specified in paper
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv_transpose(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x)
        if self.apply_dropout:
            x = self.dropout(x)
        x = self.relu(x)
        return x


class QSMUNet(nn.Module):
    """U-Net architecture for QSM reconstruction"""
    
    def __init__(self, in_channels=1, out_channels=1, filter_base=32, kernel_size=3):
        super(QSMUNet, self).__init__()
        
        self.filter_base = filter_base
        
        # Encoder (Downsampling path)
        self.down1 = DownsampleBlock(in_channels, filter_base, kernel_size, apply_batchnorm=False)
        self.down2 = DownsampleBlock(filter_base, filter_base*2, kernel_size)
        self.down3 = DownsampleBlock(filter_base*2, filter_base*3, kernel_size)
        self.down4 = DownsampleBlock(filter_base*3, filter_base*4, kernel_size)
        self.down5 = DownsampleBlock(filter_base*4, filter_base*5, kernel_size)
        
        # Decoder (Upsampling path)
        self.up1 = UpsampleBlock(filter_base*5, filter_base*5, kernel_size, apply_dropout=True)
        self.up2 = UpsampleBlock(filter_base*5 + filter_base*4, filter_base*4, kernel_size, apply_dropout=True)
        self.up3 = UpsampleBlock(filter_base*4 + filter_base*3, filter_base*3, kernel_size)
        self.up4 = UpsampleBlock(filter_base*3 + filter_base*2, filter_base*2, kernel_size)
        
        # Final layer
        self.final_conv = nn.ConvTranspose3d(
            filter_base*2 + filter_base,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size//2,
            output_padding=1
        )
        self.final_activation = nn.Tanh()
        
    def forward(self, x):
        # Encoder path with skip connections
        skip1 = self.down1(x)    # [B, 32, H/2, W/2, D/2]
        skip2 = self.down2(skip1)  # [B, 64, H/4, W/4, D/4]
        skip3 = self.down3(skip2)  # [B, 96, H/8, W/8, D/8]
        skip4 = self.down4(skip3)  # [B, 128, H/16, W/16, D/16]
        x = self.down5(skip4)      # [B, 160, H/32, W/32, D/32]
        
        # Decoder path with skip connections
        x = self.up1(x)                           # [B, 160, H/16, W/16, D/16]
        x = torch.cat([x, skip4], dim=1)          # [B, 160+128, H/16, W/16, D/16]
        x = self.up2(x)                           # [B, 128, H/8, W/8, D/8]
        x = torch.cat([x, skip3], dim=1)          # [B, 128+96, H/8, W/8, D/8]
        x = self.up3(x)                           # [B, 96, H/4, W/4, D/4]
        x = torch.cat([x, skip2], dim=1)          # [B, 96+64, H/4, W/4, D/4]
        x = self.up4(x)                           # [B, 64, H/2, W/2, D/2]
        x = torch.cat([x, skip1], dim=1)          # [B, 64+32, H/2, W/2, D/2]
        
        # Final layer
        x = self.final_conv(x)                    # [B, 1, H, W, D]
        x = self.final_activation(x)
        
        return x


class QSMModel1(QSMUNet):
    """QSM Model 1 with filter_base=32"""
    
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3):
        super(QSMModel1, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            filter_base=32,
            kernel_size=kernel_size
        )


class QSMModel2(QSMUNet):
    """QSM Model 2 with filter_base=64"""
    
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3):
        super(QSMModel2, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            filter_base=64,
            kernel_size=kernel_size
        )


def get_model(model_type="model1", **kwargs):
    """Factory function to get model by type"""
    if model_type.lower() == "model1":
        return QSMModel1(**kwargs)
    elif model_type.lower() == "model2":
        return QSMModel2(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'model1' or 'model2'")


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test input (batch_size=2, channels=1, depth=64, height=64, width=64)
    test_input = torch.randn(2, 1, 64, 64, 64).to(device)
    
    print("Testing QSM Models...")
    print(f"Input shape: {test_input.shape}")
    
    # Test Model 1
    model1 = QSMModel1().to(device)
    print(f"\nModel 1 parameters: {count_parameters(model1):,}")
    
    with torch.no_grad():
        output1 = model1(test_input)
    print(f"Model 1 output shape: {output1.shape}")
    
    # Test Model 2
    model2 = QSMModel2().to(device)
    print(f"\nModel 2 parameters: {count_parameters(model2):,}")
    
    with torch.no_grad():
        output2 = model2(test_input)
    print(f"Model 2 output shape: {output2.shape}")
    
    print("\nModels test completed successfully!")
