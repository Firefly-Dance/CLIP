# 意为不可分割的module层，用来 保存 元module

import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict


# --- ResNet Atom Module ---
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out
    
class LayerNorm(nn.LayerNorm):
    """LayerNorm but with an optional bias and automatic dtype conversion"""
    def __init__(self, ndim, bias=True):
        super().__init__(ndim, elementwise_affine=True)
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        # 确保权重和偏置的类型与输入相匹配
        weight = self.weight.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight,
            bias,
            self.eps
        )

# --- ViT Atom Module ---


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


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        # 保存输入的原始数据类型
        original_dtype = x.dtype
        
        # 如果需要，转换为浮点型进行计算
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x_float = x.float()
            result = x_float * torch.sigmoid(1.702 * x_float)
            # 转回原始数据类型
            return result.to(original_dtype)
        else:
            return x * torch.sigmoid(1.702 * x)
    

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # 获取输入的形状信息
        seq_len, batch_size, _ = x.shape
        
        # 确保权重和输入的数据类型一致
        weight_dtype = self.attn.in_proj_weight.dtype
        if x.dtype != weight_dtype:
            x = x.to(weight_dtype)
        
        # 创建因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        mask = mask.to(device=x.device, dtype=x.dtype)
        
        # 使用显式掩码，不使用 is_causal 参数
        return self.attn(
            x, x, x,
            need_weights=False,
            attn_mask=mask,
            key_padding_mask=None
        )[0]

    def forward(self, x: torch.Tensor):
        # 确保输入和权重的数据类型一致
        weight_dtype = self.ln_1.weight.dtype
        x = x.to(weight_dtype)
        
        # 应用自注意力
        x_ln = self.ln_1(x)
        attention_out = self.attention(x_ln)
        x = x + attention_out
        
        # 应用 MLP - 确保数据类型一致
        x_ln = self.ln_2(x)
        
        # 获取 MLP 的第一个线性层的权重类型
        mlp_dtype = self.mlp[0].weight.dtype
        x_ln = x_ln.to(mlp_dtype)
        
        # 应用 MLP 并确保输出与输入 x 的数据类型一致
        mlp_out = self.mlp(x_ln).to(weight_dtype)
        x = x + mlp_out
        
        return x

    def _ensure_mlp_dtype_consistency(self):
        """确保 MLP 中所有层的数据类型一致"""
        # 获取第一个线性层的数据类型
        first_dtype = next(self.mlp[0].parameters()).dtype
        
        # 确保所有层使用相同的数据类型
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.to(first_dtype)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(first_dtype)

