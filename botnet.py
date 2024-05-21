"""
Bottleneck Transformers for Visual Recognition.
adapted from https://github.com/CandiceD17/Bottleneck-Transformers-for-Visual-Recognition
"""
import torch
from einops import rearrange
from torch import einsum, nn
import os
try:
    from distribuuuu.models import resnet50
except ImportError:
    from torchvision.models import resnet50
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

## ResNet50
def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/m'
                        'odels/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # three stage spatial features (avg + std) + motion
        self.quality = self.quality_regression(4096+2048+1024, 128,1) #motion：+2048+256


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def forward(self, x):
        # See note [TorchScript super()]
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x_3D: batch x frames x (2048 + 256)
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        # x_3D: batch * frames x (2048 + 256)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x_avg2 = self.avgpool(x)
        # x_std2 = global_std_pool2d(x)

        x = self.layer3(x)
        # x_avg3 = self.avgpool(x)
        # x_std3 = global_std_pool2d(x)

        x = self.layer4(x)
        # x_avg4 = self.avgpool(x)
        # x_std4 = global_std_pool2d(x)
        # x = torch.cat((x_avg2, x_std2, x_avg3, x_std3, x_avg4, x_std4), dim=1)
        # x = torch.flatten(x, 1)

        return x

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnet34'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, inputs=(input, ))
    # print('The flops is {:.4f}, and the params is {:.4f}'.format(flops/10e9, params/10e6))
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnet50'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model

##


def expand_dim(t, dim, k):
    """
    Expand dims for t at dim to k
    """
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    """
    x: [B, Nh * H, L, 2L - 1]
    Convert relative position between the key and query to their absolute position respectively.
    Tensowflow source code in the appendix of: https://arxiv.org/pdf/1904.09925.pdf
    """
    B, Nh, L, _ = x.shape
    # pad to shift from relative to absolute indexing
    col_pad = torch.zeros((B, Nh, L, 1)).cuda()
    x = torch.cat((x, col_pad), dim=3)
    flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
    flat_pad = torch.zeros((B, Nh, L - 1)).cuda()
    flat_x = torch.cat((flat_x, flat_pad), dim=2)
    # Reshape and slice out the padded elements
    final_x = torch.reshape(flat_x, (B, Nh, L + 1, 2 * L - 1))
    return final_x[:, :, :L, L - 1 :]


def relative_logits_1d(q, rel_k):
    """
    q: [B, Nh, H, W, d]
    rel_k: [2W - 1, d]
    Computes relative logits along one dimension.
    The details of relative position is explained in: https://arxiv.org/pdf/1803.02155.pdf
    """
    B, Nh, H, W, _ = q.shape
    rel_logits = torch.einsum("b n h w d, m d -> b n h w m", q, rel_k)
    # Collapse height and heads
    rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
    rel_logits = rel_to_abs(rel_logits)
    rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
    rel_logits = expand_dim(rel_logits, dim=3, k=H)
    return rel_logits


class AbsPosEmb(nn.Module):
    def __init__(self, height, width, dim_head):
        super().__init__()
        # assert height == width
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, "h d -> h () d") + rearrange(
            self.width, "w d -> () w d"
        )
        emb = rearrange(emb, " h w d -> (h w) d")
        logits = einsum("b h i d, j d -> b h i j", q, emb)
        return logits


class RelPosEmb(nn.Module):
    def __init__(self, height, width, dim_head):
        super().__init__()
        # assert height == width
        scale = dim_head ** -0.5
        self.height = height
        self.width = width
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h = self.height
        w = self.width

        q = rearrange(q, "b h (x y) d -> b h x y d", x=h, y=w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, "b h x i y j-> b h (x y) (i j)")

        q = rearrange(q, "b h x y d -> b h y x d")
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, "b h x i y j -> b h (y x) (j i)")
        return rel_logits_w + rel_logits_h


class BoTBlock(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        dim_out,
        stride=1,
        heads=4,
        proj_factor=4,
        dim_qk=128,
        dim_v=128,
        rel_pos_emb=False,
        activation=nn.ReLU(),
    ):
        """
        dim: channels in feature map
        dim_out: output channels for feature map
        """
        super().__init__()
        if dim != dim_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(dim_out),
                activation,
            )
        else:
            self.shortcut = nn.Identity()

        bottleneck_dimension = dim_out // proj_factor  # from 2048 to 512
        attn_dim_out = heads * dim_v

        self.net = nn.Sequential(
            nn.Conv2d(dim, bottleneck_dimension, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bottleneck_dimension),
            activation,
            MHSA(
                dim=bottleneck_dimension,
                fmap_size=fmap_size,
                heads=heads,
                dim_qk=dim_qk,
                dim_v=dim_v,
                rel_pos_emb=rel_pos_emb,
            ),
            nn.AvgPool2d((2, 2)) if stride == 2 else nn.Identity(),  # same padding
            nn.BatchNorm2d(attn_dim_out),
            activation,
            nn.Conv2d(attn_dim_out, dim_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(dim_out),
        )

        nn.init.zeros_(
            self.net[-1].weight
        )  # last batch norm uses zero gamma initializer
        self.activation = activation

    def forward(self, featuremap):
        shortcut = self.shortcut(featuremap)
        featuremap = self.net(featuremap)
        featuremap += shortcut
        return self.activation(featuremap)


class MHSA(nn.Module):
    def __init__(
        self, dim, fmap_size, heads=4, dim_qk=128, dim_v=128, rel_pos_emb=False
    ):
        """
        dim: number of channels of feature map
        fmap_size: [H, W]
        dim_qk: vector dimension for q, k
        dim_v: vector dimension for v (not necessarily the same with q, k)
        """
        super().__init__()
        self.scale = dim_qk ** -0.5
        self.heads = heads
        out_channels_qk = heads * dim_qk
        out_channels_v = heads * dim_v

        self.to_qk = nn.Conv2d(
            dim, out_channels_qk * 2, 1, bias=False
        )  # 1*1 conv to compute q, k
        self.to_v = nn.Conv2d(
            dim, out_channels_v, 1, bias=False
        )  # 1*1 conv to compute v
        self.softmax = nn.Softmax(dim=-1)

        height, width = fmap_size
        if rel_pos_emb:
            self.pos_emb = RelPosEmb(height, width, dim_qk)
        else:
            self.pos_emb = AbsPosEmb(height, width, dim_qk)

    def forward(self, featuremap):
        """
        featuremap: [B, d_in, H, W]
        Output: [B, H, W, head * d_v]
        """
        heads = self.heads
        B, C, H, W = featuremap.shape
        q, k = self.to_qk(featuremap).chunk(2, dim=1)
        v = self.to_v(featuremap)
        q, k, v = map(
            lambda x: rearrange(x, "B (h d) H W -> B h (H W) d", h=heads), (q, k, v)
        )

        q = q* self.scale

        logits = einsum("b h x d, b h y d -> b h x y", q, k)
        logits += self.pos_emb(q)

        weights = self.softmax(logits)
        attn_out = einsum("b h x y, b h y d -> b h x d", weights, v)
        attn_out = rearrange(attn_out, "B h (H W) d -> B (h d) H W", H=H)

        return attn_out


class BoTStack(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        dim_out=2048,
        heads=4,
        proj_factor=4,
        num_layers=3,
        stride=2,
        dim_qk=128,
        dim_v=128,
        rel_pos_emb=False,
        activation=nn.ReLU(),
    ):
        """
        dim: channels in feature map
        fmap_size: [H, W]
        """
        super().__init__()

        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = dim if is_first else dim_out

            fmap_divisor = 2 if stride == 2 and not is_first else 1
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))

            layers.append(
                BoTBlock(
                    dim=dim,
                    fmap_size=layer_fmap_size,
                    dim_out=dim_out,
                    stride=stride if is_first else 1,
                    heads=heads,
                    proj_factor=proj_factor,
                    dim_qk=dim_qk,
                    dim_v=dim_v,
                    rel_pos_emb=rel_pos_emb,
                    activation=activation,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert c == self.dim, f"assert {c} == self.dim {self.dim}"
        # assert h == self.fmap_size[0] and w == self.fmap_size[1]
        x= self.net(x)#torch.Size([4, 1024, 28, 28])
        return x

class BotNet50(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super().__init__()
        resnet = resnet50(pretrained=False, **kwargs)
        layer = BoTStack(dim=1024, fmap_size=(14, 14), stride=1, rel_pos_emb=True)
        backbone = list(resnet.children())
        self.model = nn.Sequential(
            *backbone[:-3],
            layer,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(2048, 1000),
        )
        if pretrained:
            checkpoint = torch.load('./botnet50.pth.tar')
            self.model.load_state_dict(checkpoint)
        self.update_relative_position_bias(28)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def update_relative_position_bias(self, img_size):
        tmp_a = self.model[7].net[0].net[3].pos_emb.rel_width.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0)
        self.model[7].net[0].net[3].pos_emb.rel_width = \
            nn.Parameter(torch.nn.functional.interpolate(tmp_a, size=[img_size * 2 - 1, 128], mode='bicubic').squeeze(dim=0).squeeze(dim=0))
        tmp_a = self.model[7].net[0].net[3].pos_emb.rel_height.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0)
        self.model[7].net[0].net[3].pos_emb.rel_height = \
            nn.Parameter(torch.nn.functional.interpolate(tmp_a, size=[img_size * 2 - 1, 128], mode='bicubic').squeeze(
                dim=0).squeeze(dim=0))
        self.model[7].net[0].net[3].pos_emb.width = img_size
        self.model[7].net[0].net[3].pos_emb.height = img_size

        tmp_a = self.model[7].net[1].net[3].pos_emb.rel_width.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0)
        self.model[7].net[1].net[3].pos_emb.rel_width = \
            nn.Parameter(torch.nn.functional.interpolate(tmp_a, size=[img_size * 2 - 1, 128], mode='bicubic').squeeze(
                dim=0).squeeze(dim=0))
        tmp_a = self.model[7].net[1].net[3].pos_emb.rel_height.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0)
        self.model[7].net[1].net[3].pos_emb.rel_height = \
            nn.Parameter(torch.nn.functional.interpolate(tmp_a, size=[img_size * 2 - 1, 128], mode='bicubic').squeeze(
                dim=0).squeeze(dim=0))
        self.model[7].net[1].net[3].pos_emb.width = img_size
        self.model[7].net[1].net[3].pos_emb.height = img_size

        tmp_a = self.model[7].net[2].net[3].pos_emb.rel_width.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0)
        self.model[7].net[2].net[3].pos_emb.rel_width = \
            nn.Parameter(torch.nn.functional.interpolate(tmp_a, size=[img_size * 2 - 1, 128], mode='bicubic').squeeze(
                dim=0).squeeze(dim=0))
        tmp_a = self.model[7].net[2].net[3].pos_emb.rel_height.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0)
        self.model[7].net[2].net[3].pos_emb.rel_height = \
            nn.Parameter(torch.nn.functional.interpolate(tmp_a, size=[img_size * 2 - 1, 128], mode='bicubic').squeeze(
                dim=0).squeeze(dim=0))
        self.model[7].net[2].net[3].pos_emb.width = img_size
        self.model[7].net[2].net[3].pos_emb.height = img_size

    def forward(self, x):
        # for i in range(len(self.model)):
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        x = self.model[0](x)
        x = self.model[1](x)
        x = self.model[2](x)
        x = self.model[3](x)

        x = self.model[4](x)
        x = self.model[5](x)
        x_avg2 = self.avgpool(x)
        x_std2 = global_std_pool2d(x)

        x = self.model[6](x)
        x_avg3 = self.avgpool(x)
        x_std3 = global_std_pool2d(x)

        x = self.model[7](x)
        x_avg4 = self.avgpool(x)
        x_std4 = global_std_pool2d(x)
        x = torch.cat((x_avg2, x_std2, x_avg3, x_std3, x_avg4, x_std4), dim=1)
        x = torch.flatten(x, 1)
        return x

# def botnet50(pretrained=False, **kwargs):
#     """
#     Bottleneck Transformers for Visual Recognition.
#     https://arxiv.org/abs/2101.11605
#     """
#     resnet = resnet50(pretrained, **kwargs)
#     layer = BoTStack(dim=1024, fmap_size=(14, 14), stride=1, rel_pos_emb=True)
#     backbone = list(resnet.children())
#     model = nn.Sequential(
#         *backbone[:-3],
#         layer,
#         nn.AdaptiveAvgPool2d((1, 1)),
#         nn.Flatten(1),
#         nn.Linear(2048, 1000),
#     )
#     return model


def test_botnet50():
    x = torch.ones(2, 2, 3, 448, 448).cuda()
    model = BotNet50()
    model = model.cuda()
    y = model(x)
    print(y.shape)


def test_backbone():
    x = torch.ones(16, 3, 256, 128).cuda()
    resnet = resnet50()
    layer = BoTStack(dim=1024, fmap_size=(16, 8), stride=1, rel_pos_emb=True)
    backbone = list(resnet.children())
    model = nn.Sequential(
        *backbone[:-3],
        layer,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        nn.Linear(2048, 1000),
    ).cuda()

    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    test_botnet50()