import os
import torch
import copy
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from model.hydranet import HydraFusion
from config import Config
import torch.nn.functional as F
from torch.utils.data import random_split
from collections import defaultdict
import torch.nn as nn

def split_dataset(dataset, split_ratio=0.8):
    train_len = int(len(dataset) * split_ratio)
    val_len = len(dataset) - train_len
    return random_split(dataset, [train_len, val_len])

def validate_bbox(bbox):
    if bbox is None:
        print("Invalid bbox: None")
        return None

    # Squeeze batch dimension if present
    if bbox.dim() == 3 and bbox.shape[0] == 1:
        bbox = bbox.squeeze(0)

    if bbox.numel() == 0 or bbox.dim() != 2 or bbox.shape[1] != 4:
        print("Invalid or empty bbox after squeeze.")
        return None

    # Check for negative area
    x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    invalid_mask = (x2 <= x1) | (y2 <= y1)
    if invalid_mask.any():
        print(f"Found {invalid_mask.sum().item()} invalid boxes with negative area.")
        bbox = bbox[~invalid_mask]

    if bbox.numel() == 0:
        print("All boxes were invalid after filtering.")
        return None

    return bbox

class HydraFusionDataset(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def filter_predictions_by_iou(pred_boxes, gt_boxes, threshold=0.2):
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((0, 4)), []
    filtered_boxes, indices = [], []
    for i, pbox in enumerate(pred_boxes):
        for gtbox in gt_boxes:
            if box_iou(pbox, gtbox) >= threshold:
                filtered_boxes.append(pbox)
                indices.append(i)
                break
    return np.array(filtered_boxes), indices

def box_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

def create_bev_from_lidar(xyz, intensity, bev_size=256):
    # Ensure tensors are on CPU and flattened
    xyz = xyz.cpu().numpy().reshape(-1, 3)
    intensity = intensity.cpu().numpy().reshape(-1)

    bev = np.zeros((3, bev_size, bev_size), dtype=np.float32)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Normalize and index
    x_idx = ((x - x.min()) / (x.max() - x.min() + 1e-6) * (bev_size - 1)).astype(int)
    y_idx = ((y - y.min()) / (y.max() - y.min() + 1e-6) * (bev_size - 1)).astype(int)
    x_idx = np.clip(x_idx, 0, bev_size - 1)
    y_idx = np.clip(y_idx, 0, bev_size - 1)

    # Sanity check
    assert x_idx.shape == y_idx.shape == intensity.shape == z.shape, \
        f"Shape mismatch: x_idx {x_idx.shape}, y_idx {y_idx.shape}, intensity {intensity.shape}, z {z.shape}"

    # Assign values
    bev[0, y_idx, x_idx] = z
    bev[1, y_idx, x_idx] = intensity
    np.add.at(bev[2], (y_idx, x_idx), 1)
    bev[2] = np.clip(bev[2] / max(bev[2].max(), 1e-6), 0, 1)

    return torch.from_numpy(bev).unsqueeze(0)

def create_bev_from_radar(xyz, bev_size=256):
    xyz = xyz.cpu().numpy()
    bev = np.zeros((1, bev_size, bev_size), dtype=np.float32)
    x, y = xyz[:, 0], xyz[:, 1]
    epsilon = 1e-6
    x_idx = np.clip(((x - x.min()) / (x.max() - x.min() + epsilon) * (bev_size - 1)).astype(int), 0, bev_size - 1)
    y_idx = np.clip(((y - y.min()) / (y.max() - y.min() + epsilon) * (bev_size - 1)).astype(int), 0, bev_size - 1)
    for i in range(len(x)):
        bev[0, y_idx[i], x_idx[i]] += 1
    bev[0] = np.clip(bev[0] / max(bev[0].max(), 1e-6), 0, 1)
    return torch.tensor(bev).unsqueeze(0)
def flatten_losses(loss_dict):
    flat = {}
    for k, v in loss_dict.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat[f"{k}.{subk}"] = subv
        else:
            flat[k] = v
    return flat
def select_clients_by_mcp(client_subsets, coverage_fn, max_clients):
    selected, covered, remaining = [], set(), list(range(len(client_subsets)))
    while len(selected) < max_clients and remaining:
        best_client, best_gain = None, 0
        for i in remaining:
            new_coverage = coverage_fn(client_subsets[i]) - covered
            gain = len(new_coverage)
            if gain > best_gain:
                best_gain, best_client = gain, i
        if best_client is None:
            break
        selected.append(best_client)
        covered |= coverage_fn(client_subsets[best_client])
        remaining.remove(best_client)
    return selected

def label_coverage(subset):
    labels = set()
    for i in range(len(subset)):
        sample = subset[i]
        sample_labels = sample.get("labels")
        if sample_labels is not None and hasattr(sample_labels, "tolist"):
            labels.update(sample_labels.tolist())
    return labels

def train_modality_client(model, dataloader, cfg, device, modality, epochs=3):
    model.to(device)
    model.train()

    # Freeze all parameters except this modality’s stem + branch
    for name, param in model.named_parameters():
        if modality in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    epoch_losses = []

    for epoch in range(epochs):
        epoch_loss, valid_batches = 0.0, 0

        for input_dict in dataloader:
            optimizer.zero_grad()

            # Common pre-processing
            camera_x = input_dict.get("camera")
            lidar_xyz = input_dict.get("lidar_xyz")
            lidar_intensity = input_dict.get("lidar_intensity")
            radar_xyz = input_dict.get("radar_xyz")
            bbox_2d = validate_bbox(input_dict.get("bbox_2d"))
            target_labels = input_dict.get("labels")

            if bbox_2d is None or bbox_2d.shape[0] <= 1:
                continue

            if modality == "camera":
                if camera_x.dim() == 4 and camera_x.shape[-1] == 3:
                    camera_x = camera_x.permute(0, 3, 1, 2)
                camera_x = camera_x.to(device)
                cam_y = [{'boxes': bbox_2d.to(device), 'labels': target_labels.to(device)}]
                losses, _ = model(rightcamera_x=camera_x, leftcamera_x=camera_x, cam_y=cam_y)

            elif modality == "lidar":
                bev_lidar_x = create_bev_from_lidar(lidar_xyz, lidar_intensity)
                bev_lidar_x = F.interpolate(bev_lidar_x, size=(672, 672), mode='bilinear').to(device)
                cam_y = [{'boxes': bbox_2d.to(device), 'labels': target_labels.to(device)}]
                radar_y = [{'boxes': bbox_2d.to(device), 'labels': target_labels.to(device)}]
                losses, _ = model(bev_lidar_x=bev_lidar_x, r_lidar_x=bev_lidar_x, cam_y=cam_y, radar_y=radar_y)

            elif modality == "radar":
                radar_x = create_bev_from_radar(radar_xyz)
                radar_x = F.interpolate(radar_x, size=(672, 672), mode='bilinear').to(device)
                radar_y = [{'boxes': bbox_2d.to(device), 'labels': target_labels.to(device)}]
                losses, _ = model(radar_x=[radar_x.squeeze(0)], radar_y=radar_y)

            else:
                continue

            if not losses:
                continue

            flat_losses = flatten_losses(losses)
            loss = sum(v for v in flat_losses.values())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            valid_batches += 1

        if valid_batches:
            avg_loss = epoch_loss / valid_batches
            epoch_losses.append(avg_loss)
            print(f"  [{modality}] Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    avg_client_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    return model.state_dict(), avg_client_loss


def federated_averaging(client_weights):
    avg_weights = copy.deepcopy(client_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(client_weights)):
            avg_weights[key] += client_weights[i][key]
        avg_weights[key] = avg_weights[key] / len(client_weights)
    return avg_weights

def evaluate_model_map(model, dataset, cfg, device):
    model.eval()
    pred_boxes_list, pred_scores_list, gt_boxes_list = [], [], []

    for input_dict in dataset:
        camera_x = input_dict.get("camera")
        lidar_xyz = input_dict.get("lidar_xyz")
        lidar_intensity = input_dict.get("lidar_intensity")
        radar_xyz = input_dict.get("radar_xyz")
        bbox_2d = validate_bbox(input_dict.get("bbox_2d"))
        target_labels = input_dict.get("labels")

        if bbox_2d is None or bbox_2d.shape[0] <= 1:
            continue

        if camera_x.dim() == 3:
            camera_x = camera_x.permute(2, 0, 1).unsqueeze(0)
        camera_x = camera_x.to(device)

        bev_lidar_x = create_bev_from_lidar(lidar_xyz, lidar_intensity)
        bev_lidar_x = F.interpolate(bev_lidar_x, size=(camera_x.shape[-2], camera_x.shape[-1]),mode='bilinear').to(device)

        radar_x = create_bev_from_radar(radar_xyz)
        radar_x = F.interpolate(radar_x, size=(camera_x.shape[-2], camera_x.shape[-1]), mode='bilinear').to(device)

        cam_y = [{'boxes': bbox_2d.to(device), 'labels': target_labels.to(device)}]
        radar_y = [{'boxes': bbox_2d.to(device), 'labels': target_labels.to(device)}]
        with torch.no_grad():
            output_losses, output_detections = model(
                rightcamera_x=camera_x, leftcamera_x=camera_x,
                cam_y=cam_y,
                bev_lidar_x=bev_lidar_x,
                r_lidar_x=bev_lidar_x,
                radar_x=[radar_x.squeeze(0)],
                radar_y=radar_y
            )
            _, final_detections = model.fusion_block(output_losses, output_detections, cfg.fusion_sweep)

        pred = final_detections.get('fused2', [{}])[0]
        pred_boxes = pred.get('boxes', torch.zeros((0, 4))).cpu().numpy()
        pred_scores = pred.get('scores', torch.zeros((0,))).cpu().numpy()

        # Ensure pred_boxes is 2D before indexing
        if pred_boxes.ndim == 1:
            pred_boxes = pred_boxes.reshape(1, -1)

        if pred_boxes.shape[1] == 4:
            pred_boxes[:, [0, 2]] *= camera_x.shape[-1] / 672.0
            pred_boxes[:, [1, 3]] *= camera_x.shape[-2] / 672.0
            pred_boxes[:, [1, 3]] += 150  # vertical shift

        gt_boxes = bbox_2d.clone().cpu().numpy()
        if gt_boxes.ndim == 1:
            gt_boxes = gt_boxes.reshape(1, -1)

        if gt_boxes.shape[1] == 4:
            gt_boxes[:, [0, 2]] *= camera_x.shape[-1] / 672.0
            gt_boxes[:, [1, 3]] *= camera_x.shape[-2] / 672.0

        # Append only if predictions are valid
        if pred_boxes.shape[0] > 0:
            pred_boxes_list.append(pred_boxes)
            pred_scores_list.append(pred_scores)
            gt_boxes_list.append(gt_boxes)

    return compute_map(pred_boxes_list, pred_scores_list, gt_boxes_list)

def compute_map(pred_boxes_list, pred_scores_list, gt_boxes_list, iou_threshold=0.0):
    tp, fp, total_gt = 0, 0, 0
    for pred_boxes, pred_scores, gt_boxes in zip(pred_boxes_list, pred_scores_list, gt_boxes_list):
        filtered_preds, _ = filter_predictions_by_iou(pred_boxes, gt_boxes, threshold=iou_threshold)
        tp += len(filtered_preds)
        fp += len(pred_boxes) - len(filtered_preds)
        total_gt += len(gt_boxes)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (total_gt + 1e-6)
    return precision * recall

def run_split_federated_training(pickle_path, rounds=5):
    modalities = ["camera", "lidar", "radar"]
    args = [
        '--activation', 'relu', '--dropout', '1', '--batch_size', '1', '--device', 'cuda',
        '--fusion_type', '1', '--pretrained', 'false', '--enable_radar', 'true',
        '--enable_camera', 'true', '--enable_lidar', 'true', '--enable_cam_fusion', 'false',
        '--enable_cam_lidar_fusion', 'true', '--enable_radar_lidar_fusion', 'true',
        '--use_custom_transforms', 'true', '--fusion_sweep', '0.5', '--resume', 'true'
    ]
    cfg = Config(args)
    device = cfg.device
    dataset = HydraFusionDataset(pickle_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model_path = "modality_fed_model.pth"
    global_model = HydraFusion(cfg).to(device)
    if os.path.exists(model_path):
        print("Loading previous global model...")
        global_model.load_state_dict(torch.load(model_path))
    else:
        print("Starting new model...")

    best_map = 0.0

    for round_num in range(rounds):
        print(f"\n===== Federated Round {round_num+1} =====")
        client_weights, total_losses = {}, []

        for modality in modalities:
            print(f"\nTraining {modality.upper()} client...")
            client_model = copy.deepcopy(global_model)
            weights, avg_loss = train_modality_client(client_model, dataloader, cfg, device, modality)
            client_weights[modality] = weights
            total_losses.append(avg_loss)

        # Merge modality-specific parameters into global model
        with torch.no_grad():
            for modality in modalities:
                for name, param in global_model.named_parameters():
                    if modality in name and name in client_weights[modality]:
                        param.copy_(client_weights[modality][name])

        print(f"Avg round loss: {np.mean(total_losses):.4f}")

        # Evaluate global fusion output
        val_map = evaluate_model_map(global_model, dataset, cfg, device)
        print(f"Round {round_num+1} - mAP: {val_map:.4f}")

        if val_map > best_map:
            best_map = val_map
            torch.save(global_model.state_dict(), "modality_best_model.pth")
            print(f"New best global model (mAP={best_map:.4f})")

        torch.save(global_model.state_dict(), model_path)

    print("\nTraining complete. Model saved to", model_path)

if __name__ == "__main__":
    run_split_federated_training("test.pkl", rounds=5)
