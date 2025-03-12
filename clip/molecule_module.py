# 意为可分割的module层，用来放ModuleList化的atom module
import torch
from torch import nn
from torch.functional import 
from .atom_module import AttentionPool2d,Bottleneck,ResidualAttentionBlock,LayerNorm

# 自定义张量操作模块
class TensorReshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape  # 使用元组定义目标形状
    
    def forward(self, x):
        return x.reshape(*self.shape)
    
    def __repr__(self):
        return f"TensorReshape(shape={self.shape})"
class TensorPermute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims  # 维度排列顺序
    
    def forward(self, x):
        return x.permute(*self.dims)
    
    def __repr__(self):
        return f"TensorPermute(dims={self.dims})"
class Lambda(nn.Module):
    """将任意函数转换为可序列化的 PyTorch 模块"""
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

    def __repr__(self):
        return f"Lambda(func={self.func})"



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
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    # added 
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
        # add
        grid_size = input_resolution // patch_size
        # 构建模块列表
        self.stages = nn.ModuleList([
            # Stage 0: 分块嵌入
            nn.Sequential(
                nn.Conv2d(3, width, patch_size, patch_size, bias=False),
                TensorReshape((-1, width, grid_size**2)),  # 保持 batch 维度
                TensorPermute((0, 2, 1))  # [B, grid^2, width]
            ),
            
            # Stage 1: 添加类别嵌入
            nn.Sequential(
                Lambda(lambda x: torch.cat([
                    torch.zeros(x.size(0), 1, x.size(-1), 
                               dtype=x.dtype, device=x.device), 
                    x
                ], dim=1)),  # 占位符操作
                AddClassToken(width)  # 实际添加可学习 token
            ),
            
            # Stage 2: 位置编码
            AddPositionalEncoding((grid_size**2 + 1, width)),
            
            # Stage 3: 预处理归一化
            LayerNorm(width),
            
            # Stage 4: Transformer 输入调整
            TensorPermute((1, 0, 2)),  # [N, B, C]
            
            # Stage 5: Transformer 主体
            Transformer(width, layers, heads),
            
            # Stage 6: 输出调整
            TensorPermute((1, 0, 2)),  # [B, N, C]
            
            # Stage 7: 取类别标记
            Lambda(lambda x: x[:, 0, :]),  # 核心操作
            
            # Stage 8: 后处理归一化
            LayerNorm(width),
            
            # Stage 9: 最终投影
            ConditionalProjection(width, output_dim)
        ])


    def forward(self, x: torch.Tensor):
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
            x = x @ self.proj

        return x

