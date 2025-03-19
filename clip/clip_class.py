import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, List
from collections import OrderedDict

# 导入EosExtractor类
from molecule_module import EosExtractor, Lambda

from atom_module import AttentionPool2d, Bottleneck, LayerNorm, QuickGELU
from molecule_module import ModifiedResNet, Transformer, VisionTransformer


class TransposedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 存储 weight 的形状为 (out_features, in_features)
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # 直接使用 self.weight，不再做转置
        return F.linear(x, self.weight, self.bias)


class ModularCLIP_ResNet(nn.Module):
    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int, vision_patch_size: int, context_length: int, vocab_size: int,
                 transformer_width: int, transformer_heads: int, transformer_layers: int):
        super().__init__()
        
        self.context_length = context_length
        self.vocab_size = vocab_size
        
        # 初始化视觉编码器
        vision_heads = vision_width * 32 // 64
        self.visual = ModifiedResNet(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width
        )
        
        # 为了支持分段推理，定义视觉编码器的各个阶段
        self.vision_stages = nn.ModuleList([
            # 第一阶段：初始卷积和池化
            nn.Sequential(
                self.visual.conv1,
                self.visual.bn1,
                self.visual.relu1,
                self.visual.conv2,
                self.visual.bn2,
                self.visual.relu2,
                self.visual.conv3,
                self.visual.bn3,
                self.visual.relu3,
                self.visual.avgpool
            ),
            # 第二阶段：layer1
            self.visual.layer1,
            # 第三阶段：layer2
            self.visual.layer2,
            # 第四阶段：layer3
            self.visual.layer3,
            # 第五阶段：layer4
            self.visual.layer4,
            # 第六阶段：注意力池化
            self.visual.attnpool
        ])
        
        # 构建文本处理流
        self._build_text_modules(
            context_length, vocab_size, transformer_width,
            transformer_heads, transformer_layers, embed_dim
        )
        
        # 初始化 logit_scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def _build_text_modules(self, context_length, vocab_size, transformer_width, 
                          transformer_heads, transformer_layers, embed_dim):
        """构建可分段执行的文本模块序列"""
        # 文本嵌入
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        
        # 创建Transformer
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        
        # 最终层归一化
        self.ln_final = LayerNorm(transformer_width)
        
        # 文本投影
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        # 为分段处理创建文本阶段
        self.text_stages = nn.ModuleList([
            # Stage 0: 文本嵌入和位置编码
            Lambda(lambda x: self.token_embedding(x) + self.positional_embedding),
            
            # Stage 1: 准备Transformer输入（转置）
            Lambda(lambda x: x.permute(1, 0, 2)),  # [batch, seq, dim] -> [seq, batch, dim]
            
            # Stage 2: Transformer处理
            self.transformer,
            
            # Stage 3: 转置回原始形状
            Lambda(lambda x: x.permute(1, 0, 2)),  # [seq, batch, dim] -> [batch, seq, dim]
            
            # Stage 4: 最终层归一化
            self.ln_final,
            
            # Stage 5: 提取[EOS]位置的特征 - 这里需要原始文本输入
            EosExtractor(),
            
            # Stage 6: 文本投影
            Lambda(lambda x: x @ self.text_projection)
        ])

    def encode_image(self, image, start: int = 0, end: int = None):
        """分段执行视觉处理"""
        x = image.type(self.dtype)
        end = end if end is not None else len(self.vision_stages)
        
        for stage in self.vision_stages[start:end]:
            x = stage(x)
        return x

    def encode_text(self, text, start: int = 0, end: int = None, original_text=None):
        """分段执行文本处理"""
        x = text
        end = end if end is not None else len(self.text_stages)
        
        # 保存原始文本输入，用于提取[EOS]位置
        if original_text is None:
            original_text = text
        
        for i, stage in enumerate(self.text_stages[start:end]):
            stage_idx = i + start
            
            # 对于EosExtractor阶段，传入原始文本
            if stage_idx == 5 and isinstance(stage, EosExtractor):
                x = stage(x, original_text)
            else:
                x = stage(x)
        
        return x

    def forward(self, image, text):
        # 图像特征提取
        img_features = self.encode_image(image)
        img_features = F.normalize(img_features, p=2, dim=-1)
        
        # 文本特征提取
        txt_features = self.encode_text(text)
        txt_features = F.normalize(txt_features, p=2, dim=-1)
        
        # 计算相似度
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_features @ txt_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

    def build_attention_mask(self):
        # 创建上三角掩码
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # 上三角部分为 -inf，对角线及以下为 0
        return mask

    @property
    def dtype(self):
        return next(self.visual.parameters()).dtype

    def initialize_parameters(self):
        # 文本嵌入初始化
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        # 视觉部分初始化
        if hasattr(self.visual, 'attnpool') and self.visual.attnpool is not None:
            std = self.visual.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
        
        for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)
        
        # Transformer 初始化
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        # 文本投影初始化
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)


def copy_linear_weights(tgt, src):
    if isinstance(tgt, nn.Parameter):
        if tgt.shape != src.shape:
            min_dim0 = min(tgt.shape[0], src.shape[0])
            min_dim1 = min(tgt.shape[1], src.shape[1]) if len(tgt.shape) > 1 else 1
            if len(tgt.shape) > 1:
                tgt.data[:min_dim0, :min_dim1].copy_(src.data[:min_dim0, :min_dim1])
            else:
                tgt.data[:min_dim0].copy_(src.data[:min_dim0])
        else:
            tgt.data.copy_(src.data)
    else:
        if tgt.weight.shape != src.weight.shape:
            min_out_features = min(tgt.weight.shape[0], src.weight.shape[0])
            min_in_features = min(tgt.weight.shape[1], src.weight.shape[1])
            tgt.weight.data[:min_out_features, :min_in_features].copy_(
                src.weight.data[:min_out_features, :min_in_features]
            )
        else:
            tgt.weight.data.copy_(src.weight.data)
        if hasattr(tgt, 'bias') and hasattr(src, 'bias') and tgt.bias is not None and src.bias is not None:
            min_features = min(tgt.bias.shape[0], src.bias.shape[0])
            tgt.bias.data[:min_features].copy_(src.bias.data[:min_features])


def copy_embedding_weights(tgt, src):
    """复制嵌入层的权重，如果维度一致直接复制"""
    if tgt.weight.shape[1] == src.weight.shape[1]:
        tgt.weight.data.copy_(src.weight.data)
    else:
        tgt.weight.data[:, :src.weight.shape[1]].copy_(src.weight.data)


def copy_layernorm_weights(tgt, src):
    min_features = min(tgt.weight.shape[0], src.weight.shape[0])
    tgt.weight.data[:min_features].copy_(src.weight.data[:min_features])
    tgt.bias.data[:min_features].copy_(src.bias.data[:min_features])
    tgt.eps = src.eps


def copy_attention_block(tgt_block, src_block):
    copy_layernorm_weights(tgt_block.ln_1, src_block.ln_1)
    copy_layernorm_weights(tgt_block.ln_2, src_block.ln_2)
    copy_linear_weights(tgt_block.attn.in_proj_weight, src_block.attn.in_proj_weight)
    if hasattr(tgt_block.attn, 'in_proj_bias') and hasattr(src_block.attn, 'in_proj_bias'):
        tgt_bias = tgt_block.attn.in_proj_bias
        src_bias = src_block.attn.in_proj_bias
        min_size = min(tgt_bias.shape[0], src_bias.shape[0])
        tgt_bias.data[:min_size].copy_(src_bias.data[:min_size])
        if tgt_bias.shape[0] > min_size:
            tgt_bias.data[min_size:].zero_()
    copy_linear_weights(tgt_block.attn.out_proj, src_block.attn.out_proj)
    copy_linear_weights(tgt_block.mlp.c_fc, src_block.mlp.c_fc)
    copy_linear_weights(tgt_block.mlp.c_proj, src_block.mlp.c_proj)


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
    
    # Transformer blocks - 使用transformer_blocks而不是transformer.resblocks
    if hasattr(original_model, 'transformer_blocks'):
        for i, block in enumerate(original_model.transformer_blocks):
            for name, param in block.state_dict().items():
                if hasattr(modified_model, 'transformer'):
                    core_state_dict[f"transformer.resblocks.{i}.{name}"] = param
                else:
                    # 对于ModularCLIP_ResNet，直接加载到text_stages[2]
                    core_state_dict[f"text_stages.2.resblocks.{i}.{name}"] = param
    
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
    
    print("Successfully loaded pretrained weights")
    return modified_model.state_dict()


class CLIP_ResNet(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        # ResNet 独有的部分 start
        vision_heads = vision_width * 32 // 64
        self.visual = ModifiedResNet(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width
        )
        # ResNet 独有的部分 end

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def build_attention_mask(self):
        # 创建上三角掩码
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # 上三角部分为 -inf，对角线及以下为 0
        return mask
    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


class ModularCLIP_ViT(nn.Module):
    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: int,
                 vision_width: int, vision_patch_size: int, context_length: int, 
                 vocab_size: int, transformer_width: int, transformer_heads: int, 
                 transformer_layers: int):
        super().__init__()
        
        self.context_length = context_length
        self.vocab_size = vocab_size
        
        # 初始化视觉编码器 (ViT)
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim
        )
        
        # 初始化文本编码器组件
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # 初始化参数
        self.initialize_parameters()
        
        # 为了支持分段推理，定义视觉编码器和文本编码器的各个阶段
        # 注意：这些stages不会被保存在state_dict中，而是在每次加载模型时重新构建
        self._rebuild_stages()
    
    def _rebuild_stages(self):
        """重新构建stages，这样可以避免stages相关的键在state_dict中"""
        self.vision_stages = self._build_vision_stages()
        self.text_stages = self._build_text_stages()
    
    def _build_vision_stages(self):
        """构建视觉编码器的各个阶段，支持分段推理"""
        stages = nn.ModuleList([
            # Stage 0: 图像嵌入 (Patch + Position)
            nn.Sequential(
                # 将图像分割成补丁并嵌入
                self.visual.conv1,
                # 重塑为序列
                Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], -1)),
                Lambda(lambda x: x.permute(0, 2, 1)),
                # 添加类别标记
                Lambda(lambda x: torch.cat([
                    self.visual.class_embedding.to(x.dtype) + torch.zeros(
                        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x
                ], dim=1)),
                # 添加位置编码
                Lambda(lambda x: x + self.visual.positional_embedding.to(x.dtype)),
                # 层归一化
                self.visual.ln_pre,
                # 准备Transformer输入
                Lambda(lambda x: x.permute(1, 0, 2))  # NLD -> LND
            ),
            
            # Stage 1: Transformer (整个Transformer作为一个阶段)
            self.visual.transformer,
            
            # Stage 2: 后处理
            nn.Sequential(
                # 转换回原始形状
                Lambda(lambda x: x.permute(1, 0, 2)),  # LND -> NLD
                # 提取类别标记
                Lambda(lambda x: self.visual.ln_post(x[:, 0, :])),
                # 投影到共享空间
                Lambda(lambda x: x @ self.visual.proj if self.visual.proj is not None else x)
            )
        ])
        
        return stages
    
    def _build_text_stages(self):
        """构建文本编码器的各个阶段，支持分段推理"""
        return nn.ModuleList([
            # Stage 0: 文本嵌入和位置编码
            nn.Sequential(
                Lambda(lambda x: self.token_embedding(x) + self.positional_embedding),
                Lambda(lambda x: x.permute(1, 0, 2))  # NLD -> LND
            ),
            
            # Stage 1: Transformer (整个Transformer作为一个阶段)
            self.transformer,
            
            # Stage 2: 后处理
            nn.Sequential(
                Lambda(lambda x: x.permute(1, 0, 2)),  # LND -> NLD
                self.ln_final,
                # 提取EOS标记特征
                Lambda(lambda x, text=None: x[torch.arange(x.shape[0]), text.argmax(dim=-1)] if text is not None else x[torch.arange(x.shape[0]), x.argmax(dim=-1)]),
                # 投影到共享空间
                Lambda(lambda x: x @ self.text_projection)
            )
        ])
    
    def encode_image(self, x, start=None, end=None):
        """编码图像，支持分段处理"""
        # 确保输入和权重的数据类型匹配
        x = x.to(self.dtype)
        
        if start is None and end is None:
            # 完整处理
            return self.visual(x)
        
        # 分段处理
        for i in range(start, end):
            x = self.vision_stages[i](x)
        
        return x
    
    def encode_text(self, x, start=None, end=None, original_text=None):
        """编码文本，支持分段处理"""
        # 确保输入和权重的数据类型匹配
        x = x.to(self.dtype)
        if original_text is not None:
            original_text = original_text.to(self.dtype)
        
        if start is None and end is None:
            # 完整处理
            x = self.token_embedding(x) + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            
            # 提取EOS标记特征
            x = x[torch.arange(x.shape[0]), x.argmax(dim=-1)]
            
            # 投影到共享空间
            x = x @ self.text_projection
            
            return x
        
        # 分段处理
        if start == len(self.text_stages) - 1 and original_text is not None:  # 最后一个阶段需要原始文本
            return self.text_stages[start](x, original_text)
        else:
            return self.text_stages[start](x)
    
    def forward(self, image, text):
        """前向传播，计算图像和文本的相似度"""
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
    
    def build_attention_mask(self):
        """构建注意力掩码，用于文本Transformer"""
        # 创建上三角掩码
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # 上三角部分为 -inf，对角线及以下为 0
        return mask
    
    @property
    def dtype(self):
        return next(self.visual.parameters()).dtype
    
    def initialize_parameters(self):
        # 文本嵌入初始化
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        # 文本投影初始化
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
    
    def load_state_dict(self, state_dict, strict=False):
        """重写load_state_dict方法，在加载权重后重新构建stages"""
        result = super().load_state_dict(state_dict, strict)
        # 重新构建stages
        self._rebuild_stages()
        return result


class TextProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))
        self.bias = None
        
    def forward(self, x):
        # 确保输入和权重的数据类型一致
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        return F.linear(x, self.weight, self.bias)
