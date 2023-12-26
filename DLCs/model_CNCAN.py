## 원래 v6랑 똑같은데 간단하게 정리한 버전


#from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleOneStep(nn.Sequential):

    def __init__(self, , num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class PixelShuffleDirect(nn.Module):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        super(PixelShuffleDirect, self).__init__()
        self.upsampleOneStep = UpsampleOneStep(scale, num_feat, num_out_ch, input_resolution=None)

    def forward(self, x):
        return self.upsampleOneStep(x)


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class GCCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(GCCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace = True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
class GCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(GCALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1_ = (self.conv1(input))
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3 + cf))
        m = self.sigmoid(c4)

        return input * m

class CNCAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNCAB, self).__init__()
        self.c_L = nn.Conv2d(in_channels,out_channels , 1)
        self.gca = GCALayer(in_channels)
        self.gcca = GCCALayer(in_channels)
        self.act = nn.GELU()
    def forward(self, input) :
        L_c = self.act(self.c_L(input))
        L_C_gca = self.gca(L_c)
        L_c_gcca = self.gcca(L_c)
        L_c = L_C_gca+L_c_gcca
        return L_c

class CNCARB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNCARB, self).__init__()
        kwargs = {'padding': 1}
        self.c_R = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
        self.gca = GCALayer(in_channels)
        self.gcca = GCCALayer(in_channels)
        self.act = nn.GELU()
    def forward(self, input) :
        R_c = (self.c_R(input))
        R_c_gca = (self.gca(R_c))
        R_c_gcca = (self.gcca(R_c))
        R_c_total = R_c_gca + R_c_gcca
        R_c = self.act(R_c_total + input)
        return R_c
        
class CNCAM(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d,p=0.25):
        super(CNCAM, self).__init__()
        kwargs = {'padding': 1}

        self.c4 = conv(in_channels, out_channels, kernel_size=3, **kwargs)
        self.act = nn.GELU()
        self.CNCARB = CNCARB(in_channels, out_channels)
        self.CNCAB = CNCAB(in_channels, out_channels)

        self.c5 = nn.Conv2d(in_channels* 4, out_channels, 1)
        self.esa = ESA(in_channels, conv)
        self.gca = GCALayer(in_channels)
        self.cca = CCALayer(in_channels)

    def forward(self, input):

        L_c1 = self.CNCAB(input)
        R_c1 = self.CNCARB(input)
        L_c2 = self.CNCAB(R_c1)
        R_c2 = self.CNCARB(R_c1)
        L_c3 = self.CNCAB(R_c2)
        R_c3 = self.CNCARB(R_c2)   
        R_c4 = self.act(self.c4(R_c3))     
        
        out = torch.cat([L_c1, L_c2, L_c3, R_c4], dim=1)
        out = self.c5(out)
        out_fused = self.esa(out)
        out_fused = self.cca(out_fused)
        input_gca=self.gca(input)
        return out_fused + input + input_gca


class CNCAN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_block=8, num_out_ch=3, upscale=4,
                 conv= nn.Conv2d, upsampler='pixelshuffledirect', p=0.25):
        super(CNCAN, self).__init__()
        kwargs = {'padding': 1}
        self.conv = conv
        self.fea_conv = self.conv(num_in_ch * 4, num_feat, kernel_size=3, **kwargs)

        self.B1 = CNCAM(in_channels=num_feat, out_channels=num_feat, conv=conv, p=0.25)
        self.B2 = CNCAM(in_channels=num_feat, out_channels=num_feat, conv=conv, p=0.25)
        self.B3 = CNCAM(in_channels=num_feat, out_channels=num_feat, conv=conv, p=0.25)
        self.B4 = CNCAM(in_channels=num_feat, out_channels=num_feat, conv=conv, p=0.25)
        self.B5 = CNCAM(in_channels=num_feat, out_channels=num_feat, conv=conv, p=0.25)
        self.B6 = CNCAM(in_channels=num_feat, out_channels=num_feat, conv=conv, p=0.25)
        self.B7 = CNCAM(in_channels=num_feat, out_channels=num_feat, conv=conv, p=0.25)
        self.B8 = CNCAM(in_channels=num_feat, out_channels=num_feat, conv=conv, p=0.25)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)


        if upsampler == 'pixelshuffledirect':
            self.upsampler = PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))

    def forward(self, input):
        input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output
