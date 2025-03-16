import numpy as np
import torch
import clip
from tqdm import tqdm
from imagenet import *

# 加载模型并使用DataParallel
model, preprocess = clip.load("ViT-B/32")
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        texts = []
        for classname in classnames:
            texts.extend([template.format(classname) for template in templates])
        
        texts_tokenized = clip.tokenize(texts).cuda()
        batch_size = 512
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts_tokenized), batch_size)):
            batch = texts_tokenized[i:i + batch_size]
            with torch.amp.autocast("cuda"):
                batch_embeddings = model.encode_text(batch)
            batch_embeddings /= batch_embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(batch_embeddings.cpu())  # 减少显存占用
        
        all_embeddings = torch.cat(all_embeddings, dim=0).cuda()
        num_classes, num_templates = len(classnames), len(templates)
        embeddings_per_class = all_embeddings.view(num_classes, num_templates, -1)
        
        class_embeddings = embeddings_per_class.mean(dim=1)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        zeroshot_weights = class_embeddings.t().cuda()
    
    return zeroshot_weights

# 加载数据集
from ImagNetV2_PyTorch import ImageNetV2Dataset
images = ImageNetV2Dataset(transform=preprocess)
loader = torch.utils.data.DataLoader(images, batch_size=128, num_workers=4)

# 评估函数
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0).item()) for k in topk]

# 生成分类器权重
zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)

# 评估
with torch.no_grad():
    top1, top5, n = 0., 0., 0.
    for images, target in tqdm(loader):
        images, target = images.cuda(), target.cuda()
        
        with torch.cuda.amp.autocast():
            image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ zeroshot_weights
        
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)
    
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

print(f"Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
