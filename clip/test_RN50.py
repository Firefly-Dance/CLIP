import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn as nn
import inspect

from clip_class import ModularCLIP_ResNet, CLIP_ResNet, load_pretrained_weights
import clip
from clip import tokenize
import torchvision.transforms as transforms

def preprocess_image(image_path):
    """预处理图像，使用与CLIP相同的预处理步骤"""
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

def test_clip_model(model, image, text, model_name, precomputed_features=None):
    """测试CLIP模型的输出"""
    with torch.no_grad():
        # 如果提供了预计算的特征，直接使用
        if precomputed_features is not None:
            image_features, text_features, similarity = precomputed_features
        else:
            # 获取图像特征
            image_features = model.encode_image(image)
            
            # 获取文本特征 - 确保文本已经被tokenize
            if isinstance(text[0], str):  # 如果是字符串列表，需要tokenize
                text_tokens = clip.tokenize(text).to(image.device)
                text_features = model.encode_text(text_tokens)
            else:  # 如果已经是tokenized的张量
                text_features = model.encode_text(text)
            
            # 归一化特征
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # 计算相似度
            if hasattr(model, '__call__') and callable(getattr(model, '__call__')):
                # 如果model是可调用的（有__call__方法）
                if isinstance(text[0], str):
                    similarity = model(image, text_tokens)
                else:
                    similarity = model(image, text)
            else:
                # 手动计算相似度
                similarity = model.logit_scale.exp() * image_features @ text_features.t()
            
            # 如果是元组，取第一个元素
            if isinstance(similarity, tuple):
                similarity = similarity[0]
        
        print(f"\n=== {model_name} Model Output ===")
        print(f"Image features shape: {image_features.shape}")
        print(f"Text features shape: {text_features.shape}")
        print(f"Image features norm: {image_features.norm(dim=1).mean():.3f}")
        print(f"Text features norm: {text_features.norm(dim=1).mean():.3f}")
        print(f"Image-text similarity:")
        print(similarity)
        
        # 获取最高相似度的类别
        if similarity.shape[0] == 1:
            # 如果只有一个图像，获取最高相似度的文本
            top_probs, top_indices = similarity[0].softmax(dim=0).topk(len(text))
            print("\nTop predictions:")
            for i, (index, prob) in enumerate(zip(top_indices, top_probs)):
                print(f"{i+1}. {text[index]} ({prob:.4f})")
        
        return image_features, text_features, similarity

def expand_embedding(embedding, target_dim):
    current_dim = embedding.shape[-1]
    if current_dim >= target_dim:
        return embedding[..., :target_dim]
    repeat_times = target_dim // current_dim
    remainder = target_dim % current_dim
    expanded = embedding.repeat(*([1] * (len(embedding.shape)-1)), repeat_times)
    if remainder > 0:
        remainder_features = embedding[..., :remainder] * (1.0 / repeat_times)
        expanded = torch.cat([expanded, remainder_features], dim=-1)
    return expanded

def process_features(features, original_dim, target_dim):
    if torch.isnan(features).any():
        features = torch.nan_to_num(features, 0.0)
    if features.shape[-1] != target_dim:
        features = expand_embedding(features, target_dim)
    features = F.normalize(features, p=2, dim=-1)
    return features

def test_clip_models():
    """测试原始CLIP模型和修改后的模型"""
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载原始CLIP模型
    print("\nLoading original CLIP model...")
    original_model, preprocess = clip.load("RN50", device=device)
    original_model.eval()
    
    # 初始化修改后的模型
    print("\nInitializing modified CLIP model...")
    modified_model = ModularCLIP_ResNet(
        embed_dim=1024,
        image_resolution=224,
        vision_layers=(3, 4, 6, 3),
        vision_width=64,
        vision_patch_size=None,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12
    ).to(device)
    
    # 加载预训练权重到修改后的模型
    modified_model.load_state_dict(load_pretrained_weights(modified_model, original_model))
    modified_model.eval()
    
    # 准备测试数据
    image_path = "downloaded_image.jpg"
    texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird", 
             "a photo of a fish", "a photo of a horse", "a photo of a cow",
             "a photo of a sheep", "a photo of a rabbit", "a photo of a mouse",
             "a photo of a deer", "a photo of a fox", "a photo of a bear"]
    
    # 显示测试图像信息
    try:
        from PIL import Image
        test_image = Image.open(image_path)
        print(f"\nTest image: {image_path}")
        print(f"Image size: {test_image.size}")
        print(f"Image mode: {test_image.mode}")
    except Exception as e:
        print(f"Error displaying image info: {e}")
    
    # 预处理图像和文本
    image_input = preprocess_image(image_path).to(device)
    text_tokens = clip.tokenize(texts).to(device)
    
    # 测试原始模型
    original_image_features, original_text_features, original_similarity = test_clip_model(
        original_model, image_input, texts, "Original CLIP"
    )
    
    # 测试修改后的模型 - 完整处理
    modified_image_features, modified_text_features, modified_similarity = test_clip_model(
        modified_model, image_input, texts, "Modified CLIP (Full)"
    )
    
    # 测试修改后的模型 - 分段处理
    with torch.no_grad():
        # 分段处理图像
        stage_img = image_input
        for i in range(len(modified_model.vision_stages)):
            stage_img = modified_model.encode_image(stage_img, start=i, end=i+1)
        
        # 分段处理文本
        stage_txt = text_tokens  # 使用已经tokenize的文本
        original_text = text_tokens  # 保存原始文本输入
        for i in range(len(modified_model.text_stages)):
            stage_txt = modified_model.encode_text(stage_txt, start=i, end=i+1, original_text=original_text)
        
        # 归一化特征
        stage_img = stage_img / stage_img.norm(dim=1, keepdim=True)
        stage_txt = stage_txt / stage_txt.norm(dim=1, keepdim=True)
        
        # 计算相似度
        stage_similarity = modified_model.logit_scale.exp() * stage_img @ stage_txt.t()
    
    # 测试分段处理模型 - 传递预计算的特征
    _, _, _ = test_clip_model(
        None, image_input, texts, "Modified CLIP (Staged)", 
        precomputed_features=(stage_img, stage_txt, stage_similarity)
    )
    
    # 比较结果
    print("\n=== 模型预测比较 ===")
    
    # 获取每个模型的预测结果
    def get_top_predictions(similarity):
        probs = similarity[0].softmax(dim=0)
        top_probs, top_indices = probs.topk(3)
        return [(texts[idx], prob.item()) for idx, prob in zip(top_indices, top_probs)]
    
    original_preds = get_top_predictions(original_similarity)
    modified_preds = get_top_predictions(modified_similarity)
    staged_preds = get_top_predictions(stage_similarity)
    
    print("\n原始模型 Top-3 预测:")
    for i, (text, prob) in enumerate(original_preds):
        print(f"{i+1}. {text} ({prob:.4f})")
    
    print("\n修改后模型 (完整) Top-3 预测:")
    for i, (text, prob) in enumerate(modified_preds):
        print(f"{i+1}. {text} ({prob:.4f})")
    
    print("\n修改后模型 (分段) Top-3 预测:")
    for i, (text, prob) in enumerate(staged_preds):
        print(f"{i+1}. {text} ({prob:.4f})")
    
    # 检查预测一致性
    if original_preds[0][0] == modified_preds[0][0] == staged_preds[0][0]:
        print("\n✓ 所有模型的首选预测一致!")
    else:
        print("\n✗ 模型预测不一致!")
        print(f"原始模型: {original_preds[0][0]}")
        print(f"修改后模型 (完整): {modified_preds[0][0]}")
        print(f"修改后模型 (分段): {staged_preds[0][0]}")
    
    # 计算特征相似度
    img_sim_full = F.cosine_similarity(original_image_features.float(), modified_image_features.float())
    txt_sim_full = F.cosine_similarity(original_text_features.float(), modified_text_features.float(), dim=1)
    
    img_sim_stage = F.cosine_similarity(original_image_features.float(), stage_img.float())
    txt_sim_stage = F.cosine_similarity(original_text_features.float(), stage_txt.float(), dim=1)
    
    print("\n特征相似度:")
    print(f"图像特征 (完整): {img_sim_full.item():.6f}")
    print(f"文本特征 (完整): {txt_sim_full.mean().item():.6f}")
    print(f"图像特征 (分段): {img_sim_stage.item():.6f}")
    print(f"文本特征 (分段): {txt_sim_stage.mean().item():.6f}")

if __name__ == "__main__":
    test_clip_models()
