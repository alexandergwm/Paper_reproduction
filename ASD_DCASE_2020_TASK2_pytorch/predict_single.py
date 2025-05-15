import torch
import numpy as np
from pathlib import Path
import sys
from parameters import params
from extract_features import FeatureExtractor
from models import ASDmodel
import librosa

# 用法: python predict_single.py /path/to/audio.wav model.pth

def predict_single(audio_path, model_path, threshold=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 1. 特征提取器
    extractor = FeatureExtractor(params)
    features = extractor.extract_multiframe_features(audio_path)
    if features.shape[0] == 0:
        print("音频太短，无法提取有效特征！")
        return
    # 2. 加载模型
    feature_dim = features.shape[1]
    model = ASDmodel(input_dim=feature_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # 3. 推理
    with torch.no_grad():
        data = torch.from_numpy(features).float().to(device)
        output = model(data)
        errors = torch.mean((output - data) ** 2, dim=1).cpu().numpy()
    mean_error = np.mean(errors)
    print(f"音频: {audio_path}")
    print(f"重构误差均值: {mean_error:.6f}")
    # 4. 阈值判断
    if threshold is not None:
        result = 'anomaly' if mean_error > threshold else 'normal'
        print(f"判定结果: {result} (阈值: {threshold})")
    else:
        print("未指定阈值，仅输出重构误差。")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python predict_single.py /path/to/audio.wav model.pth [threshold]")
        sys.exit(1)
    audio_path = sys.argv[1]
    model_path = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else None
    predict_single(audio_path, model_path, threshold) 