import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
from models.models_mri import get_mri_model  # 确保你有这个函数

def load_model(model_path, device):
    model = get_mri_model('resnet18')  # 改成你实际用的模型名字
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

def load_image(path):
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 根据训练时大小调整
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

def run_inference(model, image_dir, device):
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    
    results = []
    for img_path in image_paths:
        img_tensor = load_image(img_path).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            prob = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = prob.argmax()
        results.append({
            "filename": os.path.basename(img_path),
            "predicted_class": pred_class,
            "prob_class_0": prob[0],
            "prob_class_1": prob[1]
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"⚠️ Data directory not found: {args.data_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {args.device}")
    print("Loading model...")
    model = load_model(args.model_dir, args.device)

    print("Running inference...")
    results = run_inference(model, args.data_dir, args.device)

    csv_path = os.path.join(args.output_dir, "preds_288.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"✅ Saved predictions to: {csv_path}")

if __name__ == "__main__":
    main()
