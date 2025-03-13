from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.functional as F

from .atom_module import AttentionPool2d,Bottleneck,LayerNorm,QuickGELU
from .molecule_module import ModifiedResNet,Transformer,VisionTransformer,Lambda


class ModularCLIP_ResNet(nn.Module):
    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int, vision_patch_size: int, context_length: int, vocab_size: int,
                 transformer_width: int, transformer_heads: int, transformer_layers: int):
        super().__init__()
        
        # 初始化视觉和文本的模块列表
        self.vision_stages = nn.ModuleList()
        self.text_stages = nn.ModuleList()
        
        # 构建视觉处理流
        self._build_visual_modules(
            embed_dim, image_resolution, vision_layers,
            vision_width, vision_patch_size
        )
        
        # 构建文本处理流
        self._build_text_modules(
            context_length, vocab_size, transformer_width,
            transformer_heads, transformer_layers, embed_dim
        )
        
        # 共享参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def _build_visual_modules(self, embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size):
        """构建可分段执行的视觉模块序列"""
        vision_heads = vision_width * 32 // 64
        resnet = ModifiedResNet(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width
        )
        
        # 分解ResNet为模块化组件
        self.vision_stages.extend([
            nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            ),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.attnpool if hasattr(resnet, 'attnpool') else Lambda(lambda x: x)
        ])

    def _build_text_modules(self, context_length, vocab_size, transformer_width, 
                          transformer_heads, transformer_layers, embed_dim):
        """构建可分段执行的文本模块序列"""
        self.text_stages.extend([
            # Stage 0: 文本嵌入
            nn.Sequential(
                nn.Embedding(vocab_size, transformer_width),
                Lambda(lambda x: x + self.positional_embedding)
            ),
            
            # Stage 1: Transformer输入调整
            Lambda(lambda x: x.permute(1, 0, 2)),
            
            # Stage 2: Transformer主体
            Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            ),
            
            # Stage 3: 输出调整
            Lambda(lambda x: x.permute(1, 0, 2)),
            
            # Stage 4: 最终处理
            nn.Sequential(
                LayerNorm(transformer_width),
                Lambda(lambda x: x[torch.arange(x.shape[0]), self._get_eot_positions(x)]),
                nn.Linear(transformer_width, embed_dim, bias=False)
            )
        ])
        
        # 注册文本专用参数
        self.register_parameter('positional_embedding', 
                               nn.Parameter(torch.empty(context_length, transformer_width)))
        self._eot_token_id = vocab_size - 1  # 假设最后一个token是EOS

    def _get_eot_positions(self, x):
        """动态获取每个序列的结束位置"""
        return torch.argmax(self.text_stages[0][0].weight, dim=0)[-1].expand(x.size(0))

    def encode_image(self, image, start: int = 0, end: int = None):
        """分段执行视觉处理"""
        x = image.type(self.dtype)
        end = end if end is not None else len(self.vision_stages)
        
        for stage in self.vision_stages[start:end]:
            x = stage(x)
        return x

    def encode_text(self, text, start: int = 0, end: int = None):
        """分段执行文本处理"""
        x = text
        end = end if end is not None else len(self.text_stages)
        
        for stage in self.text_stages[start:end]:
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

    # 保持原有初始化方法和属性
    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return next(self.vision_stages[0].parameters()).dtype

    def initialize_parameters(self):
        # 文本嵌入初始化
        nn.init.normal_(self.text_stages[0][0].weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        # 视觉部分初始化保持原逻辑
        for resnet_block in self.vision_stages[1:-1]:  # 跳过首尾特殊处理
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)
        
        # Transformer初始化
        text_transformer = self.text_stages[2]
        proj_std = (text_transformer.width ** -0.5) * ((2 * text_transformer.layers) ** -0.5)
        attn_std = text_transformer.width ** -0.5
        for block in text_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=attn_std)



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
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
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
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


class CLIP_ViT(nn.Module):
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

        # ViT 独有的部分 start
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )
        # ViT 独有的部分 end

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

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

