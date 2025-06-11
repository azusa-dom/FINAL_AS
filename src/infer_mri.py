import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def parse_args():
    parser = argparse.ArgumentParser("MRI 图像推理 + Grad-CAM")
    parser.add_argument("--img_path", type=str, required=True, help="输入图像路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型参数路径，如 best_fold1.pth")
    parser.add_argument("--class_names", nargs="+", required=True, help="类别名列表，如 0_Healthy 1_AS")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="cam_output", help="Grad-CAM 保存目录")
    return parser.parse_args()

def load_model(model_path, num_classes, device):
    model = models.resnet50(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_feats, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

def preprocess_image(img_path):
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0), img

def predict(model, img_tensor, device):
    with torch.no_grad():
        outputs = model(img_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
    return pred_idx, probs.squeeze().cpu().numpy()

def gradcam(model, img_tensor, target_layer="layer4", device="cpu"):
    img_tensor = img_tensor.to(device)
    gradients = []
    activations = []

    def save_gradients_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activations_hook(module, input, output):
        activations.append(output)

    layer = dict([*model.named_modules()])[target_layer]
    h1 = layer.register_forward_hook(save_activations_hook)
    h2 = layer.register_backward_hook(save_gradients_hook)

    model.eval()
    output = model(img_tensor)
    class_idx = output.argmax(dim=1).item()
    score = output[0, class_idx]
    model.zero_grad()
    score.backward()

    grad = gradients[0][0].cpu().numpy()
    act = activations[0][0].cpu().numpy()

    weights = grad.mean(axis=(1, 2))  # GAP
    cam = np.sum(act * weights[:, np.newaxis, np.newaxis], axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    h1.remove()
    h2.remove()
    return cam

def overlay_cam_on_image(img_pil, cam, alpha=0.4):
    img_np = np.array(img_pil.resize((224,224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, img_np, 1 - alpha, 0)
    return overlay

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    img_tensor, original_img = preprocess_image(args.img_path)
    model = load_model(args.model_path, len(args.class_names), device)
    pred_idx, probs = predict(model, img_tensor, device)
    pred_class = args.class_names[pred_idx]
    pred_prob = probs[pred_idx]

    print(f"✅ Prediction: {pred_class}  (Prob: {pred_prob:.4f})")

    # Grad-CAM
    cam = gradcam(model, img_tensor, device=device)
    cam_overlay = overlay_cam_on_image(original_img, cam)
    out_path = os.path.join(args.save_dir,
        f"{os.path.basename(args.img_path).split('.')[0]}_cam.jpg")
    cv2.imwrite(out_path, cam_overlay[..., ::-1])  # RGB → BGR
    print(f"✅ Grad-CAM saved to: {out_path}")

if __name__ == "__main__":
    main()
