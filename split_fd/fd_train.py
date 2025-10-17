import torch
import numpy as np
from PIL import Image
import cv2
import json
from config import Config
from model.hydranet import HydraFusion
from scipy.signal import find_peaks
import os
import copy
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d

args = [
    '--activation', 'relu',
    '--dropout', '0.0',
    '--batch_size', '1',
    '--device', 'cpu',
    '--pretrained', 'false',
    '--use_custom_transforms', 'false',
    '--fusion_sweep', 'false',
    '--resume', 'true',
    '--enable_rf_heatmap', 'true',
    '--enable_rf_spectrogram', 'true',
    '--enable_rf_fusion', 'true'
]
cfg = Config(args)
device = cfg.device

def load_image_tensor(path):
    try:
        img = Image.open(path).convert('RGB')
        return torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def load_waveform_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    waveform = np.array(data['ground_truth'], dtype=np.float32)
    waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-6)
    return torch.tensor(waveform).unsqueeze(0).to(device), torch.tensor([[data['gt_bpm']]], dtype=torch.float32).to(device)

def estimate_bpm(waveform, fs=150):
    waveform_np = waveform.squeeze().cpu().numpy()
    peaks, _ = find_peaks(waveform_np, distance=fs//2)
    duration_sec = len(waveform_np) / fs
    bpm = len(peaks) / duration_sec * 60
    return torch.tensor([[bpm]], dtype=torch.float32).to(device)

client_data = {
    "client_1": [("rf_heatmap.jpg", "rf_spectrogram.jpg", "ground_truth.json")]
}

checkpoint_path = "checkpoints/hydrafusion_rf_trained.pth"
global_model = HydraFusion(cfg).to(device)

if os.path.exists(checkpoint_path):
    global_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Checkpoint loaded successfully.")
else:
    print("No checkpoint found. Starting fresh.")

global_model.train()
num_rounds = 10
target_loss = 0.0
lr = 1e-4

for round_num in range(1, num_rounds + 1):
    client_models = []
    client_losses = []

    for client_id, samples in client_data.items():
        local_model = copy.deepcopy(global_model)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        local_model.train()
        local_loss = 0.0

        for rf_heat_path, rf_spectro_path, gt_path in samples:
            rf_heatmap = load_image_tensor(rf_heat_path).to(device)
            rf_spectrogram = load_image_tensor(rf_spectro_path).to(device)
            gt_waveform, gt_bpm = load_waveform_from_json(gt_path)

            print("rf_heatmap shape:", rf_heatmap.shape)
            print("rf_spectrogram shape:", rf_spectrogram.shape)

            optimizer.zero_grad()
            loss, output_detections, output = local_model(
                rf_heatmap_x=rf_heatmap,
                rf_spectrogram_x=rf_spectrogram,
                rf_y=gt_waveform,
            )
            loss = torch.nn.functional.mse_loss(output['fused_waveform'], gt_waveform)
            loss.backward()
            optimizer.step()
            local_loss += loss.item()

        client_models.append(local_model.state_dict())
        client_losses.append(local_loss / len(samples))

    avg_state_dict = copy.deepcopy(client_models[0])
    for key in avg_state_dict:
        for i in range(1, len(client_models)):
            avg_state_dict[key] += client_models[i][key]
    # Only divide if it's a floating point tensor
        if avg_state_dict[key].dtype in [torch.float32, torch.float64]:
            avg_state_dict[key] /= len(client_models)


    global_model.load_state_dict(avg_state_dict)
    round_loss = sum(client_losses) / len(client_losses)
    print(f"[Round {round_num}] Global Loss: {round_loss:.4f}")
    if round_loss < target_loss:
        print("Target loss reached. Stopping training.")
        break

global_model.eval()
rf_heatmap = load_image_tensor("rf_heatmap.jpg").to(device)
rf_spectrogram = load_image_tensor("rf_spectrogram.jpg").to(device)
gt_waveform, gt_bpm = load_waveform_from_json("ground_truth.json")

with torch.no_grad():
    loss,output_detections, output = global_model(
        rf_heatmap_x=rf_heatmap,
        rf_spectrogram_x=rf_spectrogram,
        rf_y=gt_waveform,
    )

print("\n--- Fused Waveform Output ---")
print(output['fused_waveform'].detach().cpu().numpy())

print("\n--- RF Fused Feature Map ---")
print(output_detections['rf_fusion'].detach().cpu().numpy())

def normalize(w): return (w - np.mean(w)) / (np.std(w) + 1e-6)
gt = normalize(gt_waveform.squeeze().cpu().numpy())
pred_raw = output['fused_waveform'].detach().cpu().squeeze().numpy()
pred_rescaled = (pred_raw - np.min(pred_raw)) / (np.max(pred_raw) - np.min(pred_raw)) * (np.max(gt) - np.min(gt)) + np.min(gt)
pred = normalize(pred_rescaled)

true_bpm = gt_bpm.item()
smoothed_pred = gaussian_filter1d(pred_rescaled, sigma=2)

def estimate_bpm(waveform, fs=150):
    waveform_np = waveform.squeeze().cpu().numpy()
    peaks, _ = find_peaks(waveform_np, distance=fs//2, prominence=0.2)
    duration_sec = len(waveform_np) / fs
    bpm = len(peaks) / duration_sec * 60
    return torch.tensor([[bpm]], dtype=torch.float32).to(device), peaks

pred_bpm_tensor, detected_peaks = estimate_bpm(torch.tensor(smoothed_pred))
pred_bpm = pred_bpm_tensor.item()

bpm_error = abs(pred_bpm - true_bpm)
bpm_accuracy = 100 - ((bpm_error / true_bpm) * 100)

rmse = np.sqrt(mean_squared_error(gt, pred))
nrmse = rmse / (np.max(gt) - np.min(gt))
waveform_accuracy = 100 - (nrmse * 100)

print(f"BPM Accuracy: {bpm_accuracy:.2f}%")
print(f"Waveform Accuracy: {waveform_accuracy:.2f}%")

try:
    import matplotlib.pyplot as plt
    time_axis = np.linspace(0, 60, num=len(smoothed_pred))
    # Plot smoothed prediction vs ground truth
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, smoothed_pred, color='red', label='Prediction')
    plt.plot(time_axis, gt, color='blue', label='Ground Truth')
    plt.title("Prediction vs Ground Truth Respiration Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("waveform_projection.jpg")
    plt.close()
    print("Waveform projection saved as waveform_projection.jpg")

except Exception as e:
    print(f"Visualization failed: {e}")

os.makedirs("checkpoints", exist_ok=True)
torch.save(global_model.state_dict(), "checkpoints/hydrafusion_rf_trained.pth")
print("Model saved.")
