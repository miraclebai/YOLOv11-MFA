# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import  LightConv
from .transformer import TransformerBlock
from ..modules.block import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from ..modules.conv import Conv, DWConv, RepConv, GhostConv, autopad
from einops import rearrange

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "C3k",
    "C3k2_DRB",
    "SF",
    "CGAFusion",
    # customized modules
    "ViTBlock",
    "ViTTransformerBlock",
    "MultiHeadAttention",
    "C2PSA_improved",
    "SF_Learnable",
    "CGAFusion_improved",
    "SpatialAttention_CGA_improved",
    "MSCAF",
"MSCAF2",
"iAFF"
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))







from ..backbone.UniRepLKNet import get_bn, get_conv2d, NCHWtoNHWC, GRNwithNHWC, SEBlock, NHWCtoNCHW, fuse_bn, merge_dilated_into_large_kernel
from timm.models.layers import DropPath
class DilatedReparamBlock(nn.Module):

    def __init__(self, channels, kernel_size, deploy=False, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def switch_to_deploy(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class UniRepLKNetBlock(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 deploy=False,
                 attempt_use_lk_impl=True,
                 with_cp=False,
                 use_sync_bn=False,
                 ffn_factor=4):
        super().__init__()
        self.with_cp = with_cp

        self.need_contiguous = (not deploy) or kernel_size >= 7

        if kernel_size == 0:
            self.dwconv = nn.Identity()
            self.norm = nn.Identity()
        elif deploy:
            self.dwconv = get_conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=True,
                                     attempt_use_lk_impl=attempt_use_lk_impl)
            self.norm = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              use_sync_bn=use_sync_bn,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
        elif kernel_size == 1:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=1, bias=deploy)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
        else:
            assert kernel_size in [3, 5]
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=dim, bias=deploy)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)

        self.se = SEBlock(dim, dim // 4)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim),
                NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                get_bn(dim, use_sync_bn=use_sync_bn))

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if (not deploy) and layer_scale_init_value is not None \
                                                         and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, inputs):

        def _f(x):
            if self.need_contiguous:
                x = x.contiguous()
            y = self.se(self.norm(self.dwconv(x)))
            y = self.pwconv2(self.act(self.pwconv1(y)))
            if self.gamma is not None:
                y = self.gamma.view(1, -1, 1, 1) * y
            return self.drop_path(y) + x

        if self.with_cp and inputs.requires_grad:
            return checkpoint.checkpoint(_f, inputs)
        else:
            return _f(inputs)

    def switch_to_deploy(self):
        if hasattr(self.dwconv, 'switch_to_deploy'):
            self.dwconv.switch_to_deploy()
        if hasattr(self.norm, 'running_var') and hasattr(self.dwconv, 'lk_origin'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
            self.dwconv.lk_origin.bias.data = self.norm.bias + (self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            self.norm = nn.Identity()
        if self.gamma is not None:
            final_scale = self.gamma.data
            self.gamma = None
        else:
            final_scale = 1
        if self.act[1].use_bias and len(self.pwconv2) == 3:
            grn_bias = self.act[1].beta.data
            self.act[1].__delattr__('beta')
            self.act[1].use_bias = False
            linear = self.pwconv2[0]
            grn_bias_projected_bias = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
            bn = self.pwconv2[2]
            std = (bn.running_var + bn.eps).sqrt()
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = linear.weight * (bn.weight / std * final_scale).view(-1, 1)
            linear_bias = 0 if linear.bias is None else linear.bias.data
            linear_bias += grn_bias_projected_bias
            new_linear.bias.data = (bn.bias + (linear_bias - bn.running_mean) * bn.weight / std) * final_scale
            self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])

class C3k_UniRepLKNetBlock(C3k):
    def __init__(self, c1, c2, n=1, k=7, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(UniRepLKNetBlock(c_, k) for _ in range(n)))

class C3k2_UniRepLKNetBlock(C3k2):
    def __init__(self, c1, c2, n=1, k=7, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_UniRepLKNetBlock(self.c, self.c, 2, k, shortcut, g) if c3k else UniRepLKNetBlock(self.c, k) for _ in range(n))

class Bottleneck_DRB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DilatedReparamBlock(c2, 7)

class C3k_DRB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DRB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_DRB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DRB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DRB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))



class SF(nn.Module):
    def __init__(self, channel=512, features_to_keep=32, *args, **kwargs):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.features_to_keep = features_to_keep

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        mean_y = torch.mean(y, dim=0, keepdim=True)
        _, indices = torch.topk(mean_y, min(self.features_to_keep, c), dim=1)
        indices = indices.repeat(b, 1)
        reduced_features = torch.gather(x, 1, indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w))

        return reduced_features


class SpatialAttention_CGA(nn.Module):
    def __init__(self):
        super(SpatialAttention_CGA, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention_CGA(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention_CGA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention_CGA(nn.Module):
    def __init__(self, dim):
        super(PixelAttention_CGA, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)
        pattn1 = pattn1.unsqueeze(dim=2)
        x2 = torch.cat([x, pattn1], dim=2)
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(dim, reduction)
        self.pa = PixelAttention_CGA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, y = data
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result
    

# customized modules
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Conv(dim, dim * 3, k=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Conv(dim, dim, k=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # 将空间维度展平
        x_flat = x.flatten(2).transpose(1, 2)  # B, N, C
        
        # 生成qkv
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, head_dim
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x.transpose(1, 2).reshape(B, C, H, W))
        x = self.proj_drop(x)
        
        return x


class ViTTransformerBlock(nn.Module):
    """Vision Transformer基本块"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        # 注意力后的残差连接
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        
        # MLP部分
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 将特征图转换为序列
        x_flat = x.flatten(2).transpose(1, 2)  # B, N, C
        
        # 第一个残差块：注意力
        x_norm = self.norm1(x_flat)
        x_flat = x_flat + self.drop_path(self.attn(x_norm.transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2))
        
        # 第二个残差块：MLP
        x_flat = x_flat + self.drop_path(self.mlp(self.norm2(x_flat)))
        
        # 恢复特征图形状
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return x


class ViTBlock(nn.Module):
    """Vision Transformer模块，保持输入输出形状一致"""
    def __init__(self, c1, c2, num_heads=8, num_layers=1, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.conv_in = Conv(c1, c2, k=1) if c1 != c2 else nn.Identity()
        self.c2 = c2
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.transformer_blocks = nn.ModuleList([
            ViTTransformerBlock(c2, num_heads, mlp_ratio, qkv_bias)
            for _ in range(num_layers)
        ])
        
        self.conv_out = Conv(c2, c2, k=1)
        
    def forward(self, x):
        x = self.conv_in(x)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.conv_out(x)
        return x
    

class C2PSA_improved(nn.Module):
    """改进的CSP Bottleneck with 2 convolutions and PSA attention"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # 增加PSABlock的数量
        self.m = nn.Sequential(*(PSABlock(c_) for _ in range(n)))
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()
        self.shortcut = shortcut and c1 == c2
        
    def forward(self, x):
        # 添加残差连接
        if self.shortcut:
            return x + self.cv3(self.m(self.cv1(x)))
        else:
            return self.cv2(x) + self.cv3(self.m(self.cv1(x)))
        

class SF_Learnable(nn.Module):
    """可学习的特征选择模块，结合平均池化和最大池化"""
    def __init__(self, c1, k=1, s=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 为平均池化和最大池化结果分别添加可学习的权重
        self.fc_avg = nn.Sequential(
            nn.Conv2d(c1, c1 // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c1 // 4, c1, 1, bias=False)
        )
        
        self.fc_max = nn.Sequential(
            nn.Conv2d(c1, c1 // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c1 // 4, c1, 1, bias=False)
        )
        
        # 融合权重
        self.sigmoid = nn.Sigmoid()
        
        # 最终的特征选择
        self.W1 = nn.Parameter(torch.ones(1))
        self.W2 = nn.Parameter(torch.ones(1))
        
        # 1x1卷积用于通道间信息交互
        self.conv = Conv(c1, c1, k=1)
        
    def forward(self, x):
        # 计算通道注意力
        avg_out = self.fc_avg(self.avg_pool(x))
        max_out = self.fc_max(self.max_pool(x))
        
        # 融合两种池化结果
        channel_att = self.sigmoid(avg_out + max_out)
        
        # 应用通道注意力
        y = channel_att * x
        
        # 最终的特征选择，使用可学习权重
        out = self.W1 * y + self.W2 * x
        
        # 1x1卷积进行通道间信息交互
        out = self.conv(out)
        
        return out
    
# 在适当位置添加以下代码

class SpatialAttention_CGA_improved(nn.Module):
    """改进的空间注意力模块，结合5x5和7x7卷积"""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 5, 7), 'kernel size must be 3, 5 or 7'
        padding = kernel_size // 2
        
        # 添加5x5卷积
        self.conv5 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        # 保留原有7x7卷积
        self.conv7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        # 1x1卷积融合两种结果
        self.conv_fusion = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿着通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # 分别使用5x5和7x7卷积
        out5 = self.conv5(x_cat)
        out7 = self.conv7(x_cat)
        
        # 融合两种结果
        out = torch.cat([out5, out7], dim=1)
        out = self.conv_fusion(out)
        
        return self.sigmoid(out)


class CGAFusion_improved(nn.Module):
    """改进的CGAFusion模块，使用改进的空间注意力"""
    def __init__(self, dim=None, reduction=8):
        super().__init__()
        self.sa = SpatialAttention_CGA_improved()
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        self.reduction = reduction
        
        # 如果初始化时提供了dim参数，则直接初始化其他组件
        if dim is not None:
            self.ca = ChannelAttention_CGA(dim, reduction)
            self.pa = PixelAttention_CGA(dim)
            self.conv = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, data):
        if isinstance(data, list):
            x, y = data
            # 确定输出通道数，优先使用第一个输入的通道数
            out_channels = x.shape[1]
            
            # 修复通道匹配检测逻辑
            if x.shape[1] != y.shape[1]:
                print(f"通道不匹配: x={x.shape[1]}, y={y.shape[1]}")
                if not hasattr(self, 'align_conv') or self.align_conv.in_channels != y.shape[1]:
                    self.align_conv = Conv(y.shape[1], out_channels, k=1).to(x.device)
                y = self.align_conv(y)
            
            # 如果没有初始化通道相关组件，现在初始化
            if not hasattr(self, 'ca') or not hasattr(self, 'pa') or not hasattr(self, 'conv'):
                self.ca = ChannelAttention_CGA(out_channels, self.reduction)
                self.pa = PixelAttention_CGA(out_channels)
                self.conv = nn.Conv2d(out_channels, out_channels, 1, bias=True)
            
            # 按照原始CGAFusion的逻辑处理
            initial = x + y
            cattn = self.ca(initial)
            sattn = self.sa(initial)
            pattn1 = sattn + cattn
            pattn2 = self.sigmoid(self.pa(initial, pattn1))
            result = initial + pattn2 * x + (1 - pattn2) * y
            result = self.conv(result)
            return result
   
    
class MSCAF(nn.Module):
    """多尺度特征融合模块，融合浅层和深层特征"""
    def __init__(self, shallow_channels, deep_channels=None, out_channels=None):
        super().__init__()
        # 如果只传入一个参数，则假设它是输出通道数
        if deep_channels is None and out_channels is None:
            out_channels = shallow_channels
            shallow_channels = None
            deep_channels = None
        
        self.out_channels = out_channels
        self.shallow_channels = shallow_channels
        self.deep_channels = deep_channels
        
        # 通道调整卷积将在forward中动态创建
        self.shallow_conv = None
        self.deep_conv = None
        
        # 空间尺寸调整
        self.down_sample = nn.MaxPool2d(kernel_size=8, stride=8)  # 320x320 -> 40x40
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 20x20 -> 40x40
        
        # 可变形注意力
        self.offset_shallow = None
        self.attention_shallow = None
        
        self.offset_deep = None
        self.attention_deep = None
        
        # 通道注意力 - 使用不含BatchNorm的卷积
        self.channel_attention = None
        
        # 输出卷积
        self.conv_out = None
        
    def _init_layers(self, shallow_channels, deep_channels, device):
        """初始化层，如果尚未初始化"""
        if self.shallow_conv is None or self.shallow_conv.conv.in_channels != shallow_channels:
            self.shallow_conv = Conv(shallow_channels, self.out_channels, k=1).to(device)
            
        if self.deep_conv is None or self.deep_conv.conv.in_channels != deep_channels:
            self.deep_conv = Conv(deep_channels, self.out_channels, k=1).to(device)
            
        if self.offset_shallow is None:
            self.offset_shallow = Conv(self.out_channels, self.out_channels, k=1).to(device)
            self.attention_shallow = nn.Sequential(
                Conv(self.out_channels, self.out_channels, k=1),
                nn.Sigmoid()
            ).to(device)
            
        if self.offset_deep is None:
            self.offset_deep = Conv(self.out_channels, self.out_channels, k=1).to(device)
            self.attention_deep = nn.Sequential(
                Conv(self.out_channels, self.out_channels, k=1),
                nn.Sigmoid()
            ).to(device)
            
        if self.channel_attention is None:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                # 使用不含BatchNorm的Conv
                nn.Conv2d(self.out_channels * 2, self.out_channels // 8, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(self.out_channels // 8, self.out_channels * 2, kernel_size=1),
                nn.Sigmoid()
            ).to(device)
            
        if self.conv_out is None:
            self.conv_out = Conv(self.out_channels * 2, self.out_channels, k=1).to(device)
        
    def forward(self, x):
        # 处理列表输入
        if isinstance(x, list):
            if len(x) == 2:
                shallow_feat, deep_feat = x
            else:
                raise ValueError(f"MSCAF需要2个输入特征，但收到了{len(x)}个")
        else:
            # 如果不是列表，则报错
            raise ValueError("MSCAF需要列表形式的输入 [shallow_feat, deep_feat]")
        
        # 初始化层
        self._init_layers(shallow_feat.shape[1], deep_feat.shape[1], shallow_feat.device)
        
        # 调整通道数
        shallow_feat = self.shallow_conv(shallow_feat)
        deep_feat = self.deep_conv(deep_feat)
        
        # 调整空间尺寸
        shallow_feat_down = self.down_sample(shallow_feat)  # 320x320 -> 40x40
        deep_feat_up = self.up_sample(deep_feat)  # 20x20 -> 40x40
        
        # 计算偏移量和注意力权重
        offset_shallow = self.offset_shallow(shallow_feat_down)
        attention_shallow = self.attention_shallow(shallow_feat_down)
        
        offset_deep = self.offset_deep(deep_feat_up)
        attention_deep = self.attention_deep(deep_feat_up)
        
        # 应用可变形注意力
        shallow_enhanced = shallow_feat_down * attention_deep + offset_deep
        deep_enhanced = deep_feat_up * attention_shallow + offset_shallow
        
        # 特征融合
        fused_feat = torch.cat([shallow_enhanced, deep_enhanced], dim=1)
        
        # 应用通道注意力
        channel_weights = self.channel_attention(fused_feat)
        fused_feat = fused_feat * channel_weights
        
        # 输出
        out = self.conv_out(fused_feat)
        
        return out
    @property
    def num_parameters(self):
        """返回模块的参数数量"""
        return sum(p.numel() for p in self.parameters())

class MSCAF2(nn.Module):
    """多尺度特征融合模块，融合浅层和深层特征"""
    def __init__(self, shallow_channels, deep_channels=None, out_channels=None):
        super().__init__()
        # 如果只传入一个参数，则假设它是输出通道数
        if deep_channels is None and out_channels is None:
            out_channels = shallow_channels
            shallow_channels = None
            deep_channels = None
        
        self.out_channels = out_channels
        self.shallow_channels = shallow_channels
        self.deep_channels = deep_channels
        
        # 直接初始化卷积层，避免动态创建
        if shallow_channels is not None and deep_channels is not None:
            self.shallow_conv = Conv(shallow_channels, out_channels, k=1)
            self.deep_conv = Conv(deep_channels, out_channels, k=1)
            
            self.offset_shallow = Conv(out_channels, out_channels, k=1)
            self.attention_shallow = nn.Sequential(
                Conv(out_channels, out_channels, k=1),
                nn.Sigmoid()
            )
            
            self.offset_deep = Conv(out_channels, out_channels, k=1)
            self.attention_deep = nn.Sequential(
                Conv(out_channels, out_channels, k=1),
                nn.Sigmoid()
            )
            
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels * 2, out_channels // 8, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(out_channels // 8, out_channels * 2, kernel_size=1),
                nn.Sigmoid()
            )
            
            self.conv_out = Conv(out_channels * 2, out_channels, k=1)
        else:
            # 如果初始化时没有提供通道数，则设为None，稍后在forward中初始化
            self.shallow_conv = None
            self.deep_conv = None
            self.offset_shallow = None
            self.attention_shallow = None
            self.offset_deep = None
            self.attention_deep = None
            self.channel_attention = None
            self.conv_out = None
        
        # 空间尺寸调整
        self.down_sample = None  # 将在forward中动态创建
        self.up_sample = None    # 将在forward中动态创建
        
    def _init_layers(self, shallow_channels, deep_channels, device):
        """初始化层，如果尚未初始化"""
        if self.shallow_conv is None:
            self.shallow_conv = Conv(shallow_channels, self.out_channels, k=1).to(device)
            
        if self.deep_conv is None:
            self.deep_conv = Conv(deep_channels, self.out_channels, k=1).to(device)
            
        if self.offset_shallow is None:
            self.offset_shallow = Conv(self.out_channels, self.out_channels, k=1).to(device)
            self.attention_shallow = nn.Sequential(
                Conv(self.out_channels, self.out_channels, k=1),
                nn.Sigmoid()
            ).to(device)
            
        if self.offset_deep is None:
            self.offset_deep = Conv(self.out_channels, self.out_channels, k=1).to(device)
            self.attention_deep = nn.Sequential(
                Conv(self.out_channels, self.out_channels, k=1),
                nn.Sigmoid()
            ).to(device)
            
        if self.channel_attention is None:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.out_channels * 2, self.out_channels // 8, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(self.out_channels // 8, self.out_channels * 2, kernel_size=1),
                nn.Sigmoid()
            ).to(device)
            
        if self.conv_out is None:
            self.conv_out = Conv(self.out_channels * 2, self.out_channels, k=1).to(device)
        
    def forward(self, x):
        # 处理列表输入
        if isinstance(x, list):
            if len(x) == 2:
                shallow_feat, deep_feat = x
            else:
                raise ValueError(f"MSCAF2需要2个输入特征，但收到了{len(x)}个")
        else:
            # 如果不是列表，则报错
            raise ValueError("MSCAF2需要列表形式的输入 [shallow_feat, deep_feat]")
        
        # 确保输出通道数已设置
        if self.out_channels is None:
            self.out_channels = min(shallow_feat.shape[1], deep_feat.shape[1])
        
        # 初始化层
        self._init_layers(shallow_feat.shape[1], deep_feat.shape[1], shallow_feat.device)
        
        # 调整通道数
        shallow_feat = self.shallow_conv(shallow_feat)
        deep_feat = self.deep_conv(deep_feat)
        
        # 获取特征图尺寸
        _, _, h_shallow, w_shallow = shallow_feat.shape
        _, _, h_deep, w_deep = deep_feat.shape
        
        # 计算目标尺寸 - 选择一个合适的中间尺寸
        target_h = min(h_shallow // 4, h_deep * 2)  # 选择较小的尺寸
        target_w = min(w_shallow // 4, w_deep * 2)  # 选择较小的尺寸
        
        # 确保目标尺寸至少为1
        target_h = max(1, target_h)
        target_w = max(1, target_w)
        
        # 动态调整空间尺寸
        import torch.nn.functional as F
        shallow_feat_down = F.interpolate(shallow_feat, size=(target_h, target_w), 
                                         mode='bilinear', align_corners=False)
        deep_feat_up = F.interpolate(deep_feat, size=(target_h, target_w), 
                                    mode='bilinear', align_corners=False)
        
        # 计算偏移量和注意力权重
        offset_shallow = self.offset_shallow(shallow_feat_down)
        attention_shallow = self.attention_shallow(shallow_feat_down)
        
        offset_deep = self.offset_deep(deep_feat_up)
        attention_deep = self.attention_deep(deep_feat_up)
        
        # 应用可变形注意力
        shallow_enhanced = shallow_feat_down * attention_deep + offset_deep
        deep_enhanced = deep_feat_up * attention_shallow + offset_shallow
        
        # 特征融合
        fused_feat = torch.cat([shallow_enhanced, deep_enhanced], dim=1)
        
        # 应用通道注意力
        channel_weights = self.channel_attention(fused_feat)
        fused_feat = fused_feat * channel_weights
        
        # 输出
        out = self.conv_out(fused_feat)
        
        return out
class iAFF_original(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力 - 修改为使用 GroupNorm 替代 BatchNorm2d
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=inter_channels),  # 使用 GroupNorm 替代 BatchNorm2d
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=channels),  # 使用 GroupNorm 替代 BatchNorm2d
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        # 第二次全局注意力 - 同样修改为使用 GroupNorm
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=inter_channels),  # 使用 GroupNorm 替代 BatchNorm2d
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=channels),  # 使用 GroupNorm 替代 BatchNorm2d
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, residual = data
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)  # 修改为使用 global_att2
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

class iAFF(nn.Module):
    '''
    multi-scale feature fusion iAFF
    '''
    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 第一次注意力（调整BN层）
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1),
            # 移除BatchNorm改用LayerNorm
            nn.GroupNorm(1, inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, 1),
            nn.GroupNorm(1, channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1),
            nn.ReLU(inplace=True),  # 移除BN层
            nn.Conv2d(inter_channels, channels, 1),
            nn.Sigmoid()  # 添加Sigmoid直接生成权重
        )

        # 第二次注意力（优化结构）
        self.pixel_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 3, padding=1, groups=inter_channels),
            nn.InstanceNorm2d(inter_channels),  # 使用InstanceNorm替代
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, 1),
            nn.Sigmoid()  # 直接生成空间注意力图
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter_channels, 1),
            nn.LayerNorm([inter_channels, 1, 1]),  # 使用LayerNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, 1),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, residual = data
        # 第一次注意力流程
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl * xg  # 改用相乘融合
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        # 第二次注意力流程
        xp = self.pixel_att(xi)
        xc = self.channel_att(xi)
        xlg2 = xp * xc  # 空间与通道注意力相乘
        wei2 = self.sigmoid(xlg2)
        
        xo = x * wei2 + residual * (1 - wei2)
        return xo