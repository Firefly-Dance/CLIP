# 意为可分割的module层，用来放ModuleList化的atom module
import torch
from torch import nn
from atom_module import AttentionPool2d,Bottleneck,ResidualAttentionBlock,LayerNorm,TensorReshape,TensorPermute
import torch.nn.functional as F
from torch.nn import MultiheadAttention

#  Helper Function
def sequential_to_modulelist(sequential: nn.Sequential) -> nn.ModuleList:
    """将 Sequential 中的子模块转换为 ModuleList
    
    Args:
        sequential (nn.Sequential): 需要转换的 Sequential 容器
        
    Returns:
        nn.ModuleList: 包含所有子模块的 ModuleList
    """
    # 从 Sequential 的有序字典中提取所有子模块
    modules = [module for _, module in sequential._modules.items()]
    return nn.ModuleList(modules)

# ResNet Module
class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)
    
    # def _make_layer_list(self, planes, blocks, stride=1):
    #     layers = [Bottleneck(self._inplanes, planes, stride)]

    #     self._inplanes = planes * Bottleneck.expansion
    #     for _ in range(1, blocks):
    #         layers.append(Bottleneck(self._inplanes, planes))

    #     return nn.ModuleList(layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x
    # added 
    def ModuleList(self):
        List = torch.nn.ModuleList()
        # stem
        List.append(self.conv1)
        List.append(self.bn1)
        List.append(self.relu1)

        List.append(self.conv2)
        List.append(self.bn2)
        List.append(self.relu2)

        List.append(self.conv3)
        List.append(self.bn3)
        List.append(self.relu3)

        List.append(self.avgpool)

        List.extend(sequential_to_modulelist(self.layer1))
        List.extend(sequential_to_modulelist(self.layer2))
        List.extend(sequential_to_modulelist(self.layer3))
        List.extend(sequential_to_modulelist(self.layer4))

        List.append(self.attnpool)


        return List


# ViT Module
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        # 确保所有块使用相同的数据类型和设备
        dtype = x.dtype
        device = x.device
        for block in self.resblocks:
            # 确保每个块的输入和权重数据类型和设备匹配
            x = block(x.to(dtype=dtype, device=device))
        return x

    def _convert_weights(self, dtype):
        """将模块的权重转换为指定的数据类型"""
        def _convert(m):
            if isinstance(m, (nn.Linear, nn.LayerNorm)):
                m.weight.data = m.weight.data.to(dtype)
                if m.bias is not None:
                    m.bias.data = m.bias.data.to(dtype)
            elif isinstance(m, nn.MultiheadAttention):
                # 转换多头注意力层的权重
                for attr in ['in_proj_weight', 'out_proj.weight']:
                    if hasattr(m, attr.split('.')[0]):
                        weight = getattr(m, attr.split('.')[0])
                        if isinstance(weight, nn.Parameter):
                            weight.data = weight.data.to(dtype)
                # 转换偏置
                for attr in ['in_proj_bias', 'out_proj.bias']:
                    if hasattr(m, attr.split('.')[0]):
                        bias = getattr(m, attr.split('.')[0])
                        if isinstance(bias, nn.Parameter):
                            bias.data = bias.data.to(dtype)
        
        self.apply(_convert)

    def ModuleList(self):
        return sequential_to_modulelist(self.resblocks)
    

# 辅助模块实现
class AddClassToken(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.token = nn.Parameter(torch.randn(width) * width**-0.5)
    
    def forward(self, x):
        # x 的形状是 [B, N, C]
        token = self.token.to(x.dtype).expand(x.size(0), 1, -1)
        return torch.cat([token, x], dim=1)
class AddPositionalEncoding(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.randn(shape) * shape[-1]**-0.5
        )
    
    def forward(self, x):
        return x + self.pos_embed.to(x.dtype)
class ConditionalProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Parameter(
            torch.randn(in_dim, out_dim) * in_dim**-0.5
        )
    
    def forward(self, x):
        return x @ self.proj.to(x.dtype)

    

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # 确保输入和权重的数据类型匹配
        dtype = self.conv1.weight.dtype
        x = x.to(dtype)
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            # 确保x和self.proj的数据类型匹配
            x = x @ self.proj.to(x.dtype)

        return x

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)

class EosExtractor(nn.Module):
    """提取[EOS]位置的特征"""
    def forward(self, x, text):
        # 找到每个序列中[EOS]标记的位置
        # 假设[EOS]是序列中的最后一个非零标记
        if text.dim() > 1:
            # 如果text是token序列，找到每个序列中最后一个非零token的位置
            eos_indices = text.ne(0).sum(dim=1) - 1
            # 确保索引不会越界
            eos_indices = torch.clamp(eos_indices, min=0, max=text.size(1)-1)
        else:
            # 如果text已经是索引，直接使用
            eos_indices = text
        
        # 提取每个序列中[EOS]位置的特征
        return x[torch.arange(x.shape[0], device=x.device), eos_indices]

class LayerNorm(nn.LayerNorm):
    """LayerNorm但具有可选的权重和偏置。不添加偏置，如果已经提供了偏置。"""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__(hidden_size, eps)
        self.weight.data.fill_(1.0)
        
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MLP(nn.Module):
    def __init__(self, d_model: int, d_out: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(d_model * 4, d_out)
    
    def forward(self, x: torch.Tensor):
        return self.c_proj(self.gelu(self.c_fc(x)))

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

