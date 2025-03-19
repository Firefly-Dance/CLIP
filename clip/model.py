from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from atom_module import LayerNorm
from molecule_module import ModifiedResNet
from clip_class import ModularCLIP_ResNet, ModularCLIP_ViT
import numpy as np


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    for param in model.parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.float16)
    
    # 对于特定的层，保持fp32精度
    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            for param_name, param in module.named_parameters():
                if param.dtype == torch.float16:
                    param.data = param.data.to(torch.float32)


def build_model(state_dict: dict):
    """根据状态字典构建CLIP模型"""
    # 检查是否为 ViT 模型的更可靠方法
    is_vit = "visual.transformer.resblocks.0.attn.in_proj_weight" in state_dict

    if is_vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    if is_vit:
        print(f"Building ViT-based model with parameters:")
        model = ModularCLIP_ViT(
            embed_dim,
            image_resolution,
            vision_layers,
            vision_width,
            vision_patch_size,
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers
        )
    else:
        print(f"Building ResNet-based model with parameters:")
        model = ModularCLIP_ResNet(
            embed_dim,
            image_resolution,
            vision_layers,
            vision_width,
            vision_patch_size,
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers
        )

    print(f"  embed_dim: {embed_dim}")
    print(f"  image_resolution: {image_resolution}")
    print(f"  vision_layers: {vision_layers}")
    print(f"  vision_width: {vision_width}")
    print(f"  vision_patch_size: {vision_patch_size}")
    print(f"  context_length: {context_length}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  transformer_width: {transformer_width}")
    print(f"  transformer_heads: {transformer_heads}")
    print(f"  transformer_layers: {transformer_layers}")

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    return model


def load_pretrained_weights(modified_model, original_model):
    """从原始CLIP模型加载预训练权重到修改后的模型"""
    print("Loading pretrained weights...")
    
    # 首先加载核心组件的权重
    core_state_dict = {}
    
    # 1. 视觉编码器权重
    for name, param in original_model.visual.state_dict().items():
        core_state_dict[f"visual.{name}"] = param
    
    # 2. 文本编码器权重
    # Token embedding
    core_state_dict["token_embedding.weight"] = original_model.token_embedding.weight
    
    # Positional embedding
    core_state_dict["positional_embedding"] = original_model.positional_embedding
    
    # Transformer blocks
    # 检查是否有transformer属性
    if hasattr(modified_model, 'transformer'):
        for i, block in enumerate(original_model.transformer.resblocks):
            for name, param in block.state_dict().items():
                core_state_dict[f"transformer.resblocks.{i}.{name}"] = param
    else:
        # 对于ModularCLIP_ResNet，直接加载到text_stages[1]
        for i, block in enumerate(original_model.transformer.resblocks):
            for name, param in block.state_dict().items():
                core_state_dict[f"text_stages.1.resblocks.{i}.{name}"] = param
    
    # 3. 最终层归一化
    core_state_dict["ln_final.weight"] = original_model.ln_final.weight
    core_state_dict["ln_final.bias"] = original_model.ln_final.bias
    
    # 4. 文本投影
    core_state_dict["text_projection"] = original_model.text_projection
    
    # 5. logit_scale
    core_state_dict["logit_scale"] = original_model.logit_scale
    
    # 加载核心组件权重
    missing_keys, unexpected_keys = modified_model.load_state_dict(core_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys in core components")
        if len(missing_keys) > 10:
            print(f"First 5 missing keys: {missing_keys[:5]}")
            print(f"Last 5 missing keys: {missing_keys[-5:]}")
    
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys in core components")
        if len(unexpected_keys) > 10:
            print(f"First 5 unexpected keys: {unexpected_keys[:5]}")
            print(f"Last 5 unexpected keys: {unexpected_keys[-5:]}")
    
    # 现在，我们将忽略stages相关的键，因为它们是从核心组件构建的
    # 这样可以避免大量的缺失键警告
    
    print("Successfully loaded pretrained weights")
    return modified_model.state_dict()


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

        if stride > 1 or inplanes != planes * self.expansion:
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
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
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

        return x[0]

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

        # 初始化权重
        self.initialize_parameters()

    def initialize_parameters(self):
        # 初始化卷积层
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 确保输入类型正确
        dtype = self.conv1.weight.dtype
        x = x.type(dtype)
        
        # 应用3层stem
        x = self.relu1(self.bn1(self.conv1(x)))  # shape: [B, width//2, 112, 112]
        x = self.relu2(self.bn2(self.conv2(x)))  # shape: [B, width//2, 112, 112]
        x = self.relu3(self.bn3(self.conv3(x)))  # shape: [B, width, 112, 112]
        x = self.avgpool(x)  # shape: [B, width, 56, 56]
        
        # 应用残差层
        x = self.layer1(x)  # shape: [B, width*4, 56, 56]
        x = self.layer2(x)  # shape: [B, width*8, 28, 28]
        x = self.layer3(x)  # shape: [B, width*16, 14, 14]
        x = self.layer4(x)  # shape: [B, width*32, 7, 7]
        
        # 应用注意力池化
        x = self.attnpool(x)  # shape: [B, output_dim]
        
        return x

# 添加缺失的ResidualAttentionBlock类
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.cross_attention = cross_attention

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ModularCLIP_ResNet(nn.Module):
    def __init__(self, embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, 
                 context_length, vocab_size, transformer_width, transformer_heads, transformer_layers):
        super().__init__()
        
        # 图像编码器
        vision_heads = vision_width * 32 // 64
        self.visual = ModifiedResNet(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width
        )
        
        # 文本编码器
        self.context_length = context_length
        self.vocab_size = vocab_size
        
        # 将文本编码器分解为多个组件
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.transformer_blocks = nn.ModuleList([
            ResidualAttentionBlock(transformer_width, transformer_heads, cross_attention=False)
            for _ in range(transformer_layers)
        ])
        self.ln_final = nn.LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        # 初始化logit_scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # 初始化参数
        self.initialize_parameters()
    
    def initialize_parameters(self):
        # 初始化positional_embedding
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        # 初始化文本编码器的其他参数
        proj_std = (self.token_embedding.weight.shape[1] ** -0.5) * ((2 * len(self.transformer_blocks)) ** -0.5)
        attn_std = self.token_embedding.weight.shape[1] ** -0.5
        fc_std = (2 * self.token_embedding.weight.shape[1]) ** -0.5
        
        for block in self.transformer_blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        # 初始化text_projection
        nn.init.normal_(self.text_projection, std=self.token_embedding.weight.shape[1] ** -0.5)
    
    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # [n_ctx, batch_size, d_model]
        
        # 应用Transformer块
        for block in self.transformer_blocks:
            x = block(x)
        
        x = x.permute(1, 0, 2)  # [batch_size, n_ctx, d_model]
        x = self.ln_final(x)  # [batch_size, n_ctx, d_model]
        
        # 取[EOS]位置的特征
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]  # [batch_size, d_model]
        
        # 应用投影
        x = x @ self.text_projection  # [batch_size, embed_dim]
        
        return x
    
    def encode_image(self, image):
        return self.visual(image)
    
    def forward(self, image, text):
        # 获取图像和文本特征
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # 归一化特征
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # 计算相似度
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
