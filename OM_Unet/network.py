import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """convolution block"""

    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DConvBlock(nn.Module):
    """double convolution block"""

    def __init__(self, ch_in, ch_out):
        super(DConvBlock, self).__init__()
        self.doubleconv = nn.Sequential(
            ConvBlock(ch_in, ch_out),
            ConvBlock(ch_out, ch_out),
        )

    def forward(self, x):
        return self.doubleconv(x)


class PreActConvBlock(nn.Module):
    """pre-activation convolution block"""

    def __init__(self, ch_in, ch_out):
        super(PreActConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.conv(x)


class DPreActBlock(nn.Module):
    """double convolution block"""

    def __init__(self, ch_in, ch_out):
        super(DPreActBlock, self).__init__()
        self.doubleconv = nn.Sequential(
            PreActConvBlock(ch_in, ch_out),
            PreActConvBlock(ch_out, ch_out),
        )

    def forward(self, x):
        return self.doubleconv(x)


class ResConvBlock(nn.Module):
    """residual convolution block"""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.shortcut = nn.Sequential()
        if ch_in != ch_out:
            self.shortcut = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = self.relu(x)
        return x


class PreActResBlock(nn.Module):
    """Pre-activation ResNet (ResNet v2)"""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.preactconv = DPreActBlock(ch_in, ch_out)
        self.shortcut = nn.Sequential()
        if ch_in != ch_out:
            self.shortcut = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.preactconv(x)
        return out + identity


class RRConvBlock(nn.Module):
    """recurrent residual convolution block"""

    def __init__(self, ch_in, ch_out, t=1):
        super(RRConvBlock, self).__init__()
        self.t = t
        self.conv1 = ConvBlock(ch_in, ch_out)
        self.conv2 = ConvBlock(ch_out, ch_out)
        self.shortcut = nn.Sequential()
        if ch_in != ch_out:
            self.shortcut = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        for _ in range(self.t):
            out = self.conv2(x)
            x = x + out
        return x + residual


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        hidden_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_att = self.mlp(self.avg_pool(x))
        max_att = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_att + max_att)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        att = self.conv(torch.cat([avg_map, max_map], dim=1))
        return x * self.sigmoid(att)


class CBAM(nn.Module):
    def __init__(self, ch_in, ch_out, reduction=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.conv = DConvBlock(ch_in, ch_out)
        self.channel_att = ChannelAttention(ch_out, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ResCBAM(nn.Module):
    def __init__(self, ch_in, ch_out, reduction=16, spatial_kernel_size=7):
        super(ResCBAM, self).__init__()
        self.preactconv = DPreActBlock(ch_in, ch_out)
        self.channel_att = ChannelAttention(ch_out, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel_size)
        self.shortcut = nn.Sequential()
        if ch_in != ch_out:
            self.shortcut = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.preactconv(x)
        out = self.channel_att(out)
        out = self.spatial_att(out)
        return out + identity


class AttGate(nn.Module):
    """attention gate"""

    def __init__(self, F_g, F_l, F_int):
        super(AttGate, self).__init__()
        # F_g: channels of gating signal (from decoder)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int),
        )
        # F_l: channels of the feature map to be attended (from encoder)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int),
        )
        # F_int: intermediate channels for computing attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Unet_2L(nn.Module):
    def __init__(self, input_ch=1, output_ch=1):
        super(Unet_2L, self).__init__()

        # ---------------- encoder ----------------
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = DConvBlock(input_ch, 64)
        self.enc2 = DConvBlock(64, 128)

        # ---------------- decoder ----------------
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_reduce = ConvBlock(128, 64)
        self.up2_fuse = DConvBlock(128, 64)

        # -------------- output layer --------------
        self.head = ConvBlock(64, 64)
        self.out_conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ---------------- encoder ----------------
        x1 = self.enc1(x)
        x2 = self.maxpool(x1)
        x2 = self.enc2(x2)

        # ---------------- decoder ----------------
        d2 = self.up2(x2)
        d2 = self.up2_reduce(d2)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.up2_fuse(d2)

        # -------------- output layer --------------
        d2 = self.head(d2)
        out = self.out_conv(d2)
        return out


class Unet_3L(nn.Module):
    def __init__(self, input_ch=1, output_ch=1):
        super(Unet_3L, self).__init__()

        # ---------------- encoder ----------------
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # max pooling
        self.enc1 = DConvBlock(input_ch, 64)
        self.enc2 = DConvBlock(64, 128)
        self.enc3 = DConvBlock(128, 256)

        # ---------------- decoder ----------------
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')  # up sampling
        self.up3_reduce = ConvBlock(256, 128)
        self.up3_fuse = DConvBlock(256, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_reduce = ConvBlock(128, 64)
        self.up2_fuse = DConvBlock(128, 64)

        # -------------- output layer --------------
        self.head = ConvBlock(64, 64)
        self.out_conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ---------------- encoder ----------------
        x1 = self.enc1(x)
        x2 = self.maxpool(x1)
        x2 = self.enc2(x2)
        x3 = self.maxpool(x2)
        x3 = self.enc3(x3)

        # ---------------- decoder ----------------
        d3 = self.up3(x3)
        d3 = self.up3_reduce(d3)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.up3_fuse(d3)

        d2 = self.up2(d3)
        d2 = self.up2_reduce(d2)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.up2_fuse(d2)

        # -------------- output layer --------------
        d2 = self.head(d2)
        out = self.out_conv(d2)
        return out


class Unet_4L(nn.Module):
    def __init__(self, input_ch=1, output_ch=1):
        super(Unet_4L, self).__init__()

        # ---------------- encoder ----------------
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = DConvBlock(input_ch, 64)
        self.enc2 = DConvBlock(64, 128)
        self.enc3 = DConvBlock(128, 256)
        self.enc4 = DConvBlock(256, 512)

        # ---------------- decoder ----------------
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4_reduce = ConvBlock(512, 256)
        self.up4_fuse = DConvBlock(512, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3_reduce = ConvBlock(256, 128)
        self.up3_fuse = DConvBlock(256, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_reduce = ConvBlock(128, 64)
        self.up2_fuse = DConvBlock(128, 64)

        # -------------- output layer --------------
        self.head = ConvBlock(64, 64)
        self.out_conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ---------------- encoder ----------------
        x1 = self.enc1(x)
        x2 = self.maxpool(x1)
        x2 = self.enc2(x2)
        x3 = self.maxpool(x2)
        x3 = self.enc3(x3)
        x4 = self.maxpool(x3)
        x4 = self.enc4(x4)

        # ---------------- decoder ----------------
        d4 = self.up4(x4)
        d4 = self.up4_reduce(d4)
        d4 = torch.cat([x3, d4], dim=1)
        d4 = self.up4_fuse(d4)

        d3 = self.up3(d4)
        d3 = self.up3_reduce(d3)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.up3_fuse(d3)

        d2 = self.up2(d3)
        d2 = self.up2_reduce(d2)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.up2_fuse(d2)

        # -------------- output layer --------------
        d2 = self.head(d2)
        out = self.out_conv(d2)
        return out


class Unet_5L(nn.Module):
    def __init__(self, input_ch=1, output_ch=1):
        super(Unet_5L, self).__init__()

        # ---------------- encoder ----------------
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = DConvBlock(input_ch, 64)
        self.enc2 = DConvBlock(64, 128)
        self.enc3 = DConvBlock(128, 256)
        self.enc4 = DConvBlock(256, 512)
        self.enc5 = DConvBlock(512, 1024)

        # ---------------- decoder ----------------
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up5_reduce = ConvBlock(1024, 512)
        self.up5_fuse = DConvBlock(1024, 512)

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4_reduce = ConvBlock(512, 256)
        self.up4_fuse = DConvBlock(512, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3_reduce = ConvBlock(256, 128)
        self.up3_fuse = DConvBlock(256, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_reduce = ConvBlock(128, 64)
        self.up2_fuse = DConvBlock(128, 64)

        # -------------- output layer --------------
        self.head = ConvBlock(64, 64)
        self.out_conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ---------------- encoder ----------------
        x1 = self.enc1(x)
        x2 = self.maxpool(x1)
        x2 = self.enc2(x2)
        x3 = self.maxpool(x2)
        x3 = self.enc3(x3)
        x4 = self.maxpool(x3)
        x4 = self.enc4(x4)
        x5 = self.maxpool(x4)
        x5 = self.enc5(x5)

        # ---------------- decoder ----------------
        d5 = self.up5(x5)
        d5 = self.up5_reduce(d5)
        d5 = torch.cat([x4, d5], dim=1)
        d5 = self.up5_fuse(d5)

        d4 = self.up4(d5)
        d4 = self.up4_reduce(d4)
        d4 = torch.cat([x3, d4], dim=1)
        d4 = self.up4_fuse(d4)

        d3 = self.up3(d4)
        d3 = self.up3_reduce(d3)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.up3_fuse(d3)

        d2 = self.up2(d3)
        d2 = self.up2_reduce(d2)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.up2_fuse(d2)

        # -------------- output layer --------------
        d2 = self.head(d2)
        out = self.out_conv(d2)
        return out


class Res_Unet_4L(nn.Module):
    def __init__(self, input_ch=1, output_ch=1):
        super(Res_Unet_4L, self).__init__()

        # ---------------- encoder ----------------
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res1 = PreActResBlock(input_ch, 64)
        self.res2 = PreActResBlock(64, 128)
        self.res3 = PreActResBlock(128, 256)
        self.res4 = PreActResBlock(256, 512)

        # ---------------- decoder ----------------
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4_reduce = ConvBlock(512, 256)
        self.up4_fuse = DConvBlock(512, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3_reduce = ConvBlock(256, 128)
        self.up3_fuse = DConvBlock(256, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_reduce = ConvBlock(128, 64)
        self.up2_fuse = DConvBlock(128, 64)

        # -------------- output layer --------------
        self.head = ConvBlock(64, 64)
        self.out_conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ---------------- encoder ----------------
        x1 = self.res1(x)
        x2 = self.maxpool(x1)
        x2 = self.res2(x2)
        x3 = self.maxpool(x2)
        x3 = self.res3(x3)
        x4 = self.maxpool(x3)
        x4 = self.res4(x4)

        # ---------------- decoder ----------------
        d4 = self.up4(x4)
        d4 = self.up4_reduce(d4)
        d4 = torch.cat([x3, d4], dim=1)
        d4 = self.up4_fuse(d4)

        d3 = self.up3(d4)
        d3 = self.up3_reduce(d3)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.up3_fuse(d3)

        d2 = self.up2(d3)
        d2 = self.up2_reduce(d2)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.up2_fuse(d2)

        # -------------- output layer --------------
        d2 = self.head(d2)
        out = self.out_conv(d2)
        return out


class CBAM_Unet_4L(nn.Module):
    def __init__(self, input_ch=1, output_ch=1):
        super(CBAM_Unet_4L, self).__init__()

        # ---------------- encoder ----------------
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = CBAM(input_ch, 64)
        self.enc2 = CBAM(64, 128)
        self.enc3 = CBAM(128, 256)
        self.enc4 = CBAM(256, 512)

        # ---------------- decoder ----------------
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4_reduce = ConvBlock(512, 256)
        self.up4_fuse = DConvBlock(512, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3_reduce = ConvBlock(256, 128)
        self.up3_fuse = DConvBlock(256, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_reduce = ConvBlock(128, 64)
        self.up2_fuse = DConvBlock(128, 64)

        # -------------- output layer --------------
        self.head = ConvBlock(64, 64)
        self.out_conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ---------------- encoder ----------------
        x1 = self.enc1(x)
        x2 = self.maxpool(x1)
        x2 = self.enc2(x2)
        x3 = self.maxpool(x2)
        x3 = self.enc3(x3)
        x4 = self.maxpool(x3)
        x4 = self.enc4(x4)

        # ---------------- decoder ----------------
        d4 = self.up4(x4)
        d4 = self.up4_reduce(d4)
        d4 = torch.cat([x3, d4], dim=1)
        d4 = self.up4_fuse(d4)

        d3 = self.up3(d4)
        d3 = self.up3_reduce(d3)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.up3_fuse(d3)

        d2 = self.up2(d3)
        d2 = self.up2_reduce(d2)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.up2_fuse(d2)

        # -------------- output layer --------------
        d2 = self.head(d2)
        out = self.out_conv(d2)
        return out


class ResCBAM_Unet_4L(nn.Module):
    def __init__(self, input_ch=1, output_ch=1):
        super(ResCBAM_Unet_4L, self).__init__()

        # ---------------- encoder ----------------
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = ResCBAM(input_ch, 64)
        self.enc2 = ResCBAM(64, 128)
        self.enc3 = ResCBAM(128, 256)
        self.enc4 = ResCBAM(256, 512)

        # ---------------- decoder ----------------
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4_reduce = ConvBlock(512, 256)
        self.up4_fuse = DConvBlock(512, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3_reduce = ConvBlock(256, 128)
        self.up3_fuse = DConvBlock(256, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_reduce = ConvBlock(128, 64)
        self.up2_fuse = DConvBlock(128, 64)

        # -------------- output layer --------------
        self.head = ConvBlock(64, 64)
        self.out_conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ---------------- encoder ----------------
        x1 = self.enc1(x)
        x2 = self.maxpool(x1)
        x2 = self.enc2(x2)
        x3 = self.maxpool(x2)
        x3 = self.enc3(x3)
        x4 = self.maxpool(x3)
        x4 = self.enc4(x4)

        # ---------------- decoder ----------------
        d4 = self.up4(x4)
        d4 = self.up4_reduce(d4)
        d4 = torch.cat([x3, d4], dim=1)
        d4 = self.up4_fuse(d4)

        d3 = self.up3(d4)
        d3 = self.up3_reduce(d3)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.up3_fuse(d3)

        d2 = self.up2(d3)
        d2 = self.up2_reduce(d2)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.up2_fuse(d2)

        # -------------- output layer --------------
        d2 = self.head(d2)
        out = self.out_conv(d2)
        return out


class LAtt_Unet_4L(nn.Module):
    def __init__(self, input_ch=1, output_ch=1):
        super(LAtt_Unet_4L, self).__init__()

        # ---------------- encoder ----------------
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = DConvBlock(input_ch, 64)
        self.enc2 = DConvBlock(64, 128)
        self.enc3 = DConvBlock(128, 256)
        self.enc4 = DConvBlock(256, 512)

        # ---------------- decoder ----------------
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4_reduce = ConvBlock(512, 256)
        self.AG4 = AttGate(F_g=256, F_l=256, F_int=64)
        self.up4_fuse = DConvBlock(512, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3_reduce = ConvBlock(256, 128)
        self.up3_fuse = DConvBlock(256, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_reduce = ConvBlock(128, 64)
        self.up2_fuse = DConvBlock(128, 64)

        # -------------- output layer --------------
        self.head = ConvBlock(64, 64)
        self.out_conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ---------------- encoder ----------------
        x1 = self.enc1(x)
        x2 = self.maxpool(x1)
        x2 = self.enc2(x2)
        x3 = self.maxpool(x2)
        x3 = self.enc3(x3)
        x4 = self.maxpool(x3)
        x4 = self.enc4(x4)

        # ---------------- decoder ----------------
        d4 = self.up4(x4)
        d4 = self.up4_reduce(d4)
        x3_att = self.AG4(g=d4, x=x3)
        d4 = torch.cat([x3_att, d4], dim=1)
        d4 = self.up4_fuse(d4)

        d3 = self.up3(d4)
        d3 = self.up3_reduce(d3)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.up3_fuse(d3)

        d2 = self.up2(d3)
        d2 = self.up2_reduce(d2)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.up2_fuse(d2)

        # -------------- output layer --------------
        d2 = self.head(d2)
        out = self.out_conv(d2)
        return out


class R2_Unet_4L(nn.Module):
    def __init__(self, input_ch=1, output_ch=1):
        super(R2_Unet_4L, self).__init__()

        # ---------------- encoder ----------------
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res1 = RRConvBlock(input_ch, 64)
        self.res2 = RRConvBlock(64, 128)
        self.res3 = RRConvBlock(128, 256)
        self.res4 = RRConvBlock(256, 512)

        # ---------------- decoder ----------------
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4_reduce = ConvBlock(512, 256)
        self.up4_fuse = RRConvBlock(512, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3_reduce = ConvBlock(256, 128)
        self.up3_fuse = RRConvBlock(256, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_reduce = ConvBlock(128, 64)
        self.up2_fuse = RRConvBlock(128, 64)

        # -------------- output layer --------------
        self.head = ConvBlock(64, 64)
        self.out_conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ---------------- encoder ----------------
        x1 = self.res1(x)
        x2 = self.maxpool(x1)
        x2 = self.res2(x2)
        x3 = self.maxpool(x2)
        x3 = self.res3(x3)
        x4 = self.maxpool(x3)
        x4 = self.res4(x4)

        # ---------------- decoder ----------------
        d4 = self.up4(x4)
        d4 = self.up4_reduce(d4)
        d4 = torch.cat([x3, d4], dim=1)
        d4 = self.up4_fuse(d4)

        d3 = self.up3(d4)
        d3 = self.up3_reduce(d3)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.up3_fuse(d3)

        d2 = self.up2(d3)
        d2 = self.up2_reduce(d2)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.up2_fuse(d2)

        # -------------- output layer --------------
        d2 = self.head(d2)
        out = self.out_conv(d2)
        return out


def build_model(config, device, use_compile=False, use_channels_last=False):
    input_ch = sum(len(detail['vars']) for name, detail in config.ds_info.items() if name != config.target_ds)
    output_ch = len(config.ds_info[config.target_ds]['vars'])

    if config.model_type == 'Unet_2L':
        model = Unet_2L(input_ch, output_ch)
    elif config.model_type == 'Unet_3L':
        model = Unet_3L(input_ch, output_ch)
    elif config.model_type == 'Unet_4L':
        model = Unet_4L(input_ch, output_ch)
    elif config.model_type == 'Unet_5L':
        model = Unet_5L(input_ch, output_ch)
    elif config.model_type == 'Res_Unet_4L':
        model = Res_Unet_4L(input_ch, output_ch)
    elif config.model_type == 'CBAM_Unet_4L':
        model = CBAM_Unet_4L(input_ch, output_ch)
    elif config.model_type == 'ResCBAM_Unet_4L':
        model = ResCBAM_Unet_4L(input_ch, output_ch)
    elif config.model_type == 'LAtt_Unet_4L':
        model = LAtt_Unet_4L(input_ch, output_ch)
    elif config.model_type == 'R2_Unet_4L':
        model = R2_Unet_4L(input_ch, output_ch)
    else:
        raise NotImplementedError(f'Model type {config.model_type} not supported!')

    memory_format = torch.channels_last if use_channels_last else torch.preserve_format
    model = model.to(device=device, memory_format=memory_format)

    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='default')
        except Exception as exc:
            print(f'torch.compile failed, fallback to eager mode: {exc}')

    return model


if __name__ == '__main__':
    import config_loader

    build_model(config_loader, 'cuda')
