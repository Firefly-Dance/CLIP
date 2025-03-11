import clip
from PIL import Image
import torch
import warnings

# model_name_list = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
# for model_name in model_name_list:

if True:
    model_name = 'RN50'
    warnings.warn(f"Using model {model_name}")
    device = "cuda:0"

    model, preprocess = clip.load(model_name)
    model.cuda().eval()

    image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print(probs)