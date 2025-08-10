import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
from efficientnet_pytorch import EfficientNet

# === Label setup ===
json_path = r"C:\Users\user\Desktop\TEKNOFEST_PUBLIC\MR_Yeni (1) (1).json"
with open(json_path, "r") as f:
    content = f.read()
    if not content.strip().startswith("["):
        content = "[" + content.strip().rstrip(",") + "]"
    data = json.loads(content)

LABELS = ["HiperakutAkut", "Subakut", "NormalKronik"]
label_map = {label: idx for idx, label in enumerate(LABELS)}

patient_label_map = defaultdict(set)
for item in data:
    pid = item["PatientId"]
    label = item["LessionTypeName"]
    patient_label_map[pid].add(label)

final_label_dict = {}
for pid, labels in patient_label_map.items():
    one_hot = [0] * len(LABELS)
    for lbl in labels:
        if lbl in label_map:
            one_hot[label_map[lbl]] = 1
    final_label_dict[pid] = one_hot

# === Dataset ===
class MultiModalDataset(Dataset):
    def __init__(self, mri_root, ct_root, label_dict, transform=None):
        self.mri_root = mri_root
        self.ct_root = ct_root
        self.label_dict = label_dict
        self.transform = transform
        self.patient_ids = [pid for pid in os.listdir(mri_root)
                            if os.path.isdir(os.path.join(mri_root, pid)) and pid in label_dict]

    def __len__(self):
        return len(self.patient_ids)

    def load_npy_stack(self, folder):
        if not os.path.exists(folder): return None
        npy_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
        if not npy_files: return None
        images = [np.load(f) for f in npy_files]
        image = np.mean(np.stack(images), axis=0)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        image = torch.tensor(image).float() / 255.0
        image = image.permute(2, 0, 1)
        return image

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        mri_folder = os.path.join(self.mri_root, pid)
        ct_folder = os.path.join(self.ct_root, pid, "CT")

        mri_img = self.load_npy_stack(mri_folder)
        ct_img = self.load_npy_stack(ct_folder)

        if mri_img is None and ct_img is None:
            return None

        if self.transform:
            if mri_img is not None:
                mri_img = self.transform(mri_img)
            if ct_img is not None:
                ct_img = self.transform(ct_img)

        label = torch.tensor(self.label_dict.get(pid, [0]*len(LABELS)), dtype=torch.float32)
        return mri_img, ct_img, label, pid

# === Collate function ===
def custom_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    mri_list, ct_list, labels, pids = zip(*batch)
    labels = torch.stack(labels)

    def fill_missing(modality_list):
        return torch.stack([
            img if img is not None else torch.zeros((3, 224, 224))
            for img in modality_list
        ])

    mri = fill_missing(mri_list)
    ct = fill_missing(ct_list)
    return mri, ct, labels, pids

# === Model ===
class MultiModalEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mri_backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.ct_backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.mri_backbone._fc = nn.Identity()
        self.ct_backbone._fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, mri, ct):
        features = []
        if mri is not None:
            features.append(self.mri_backbone(mri))
        if ct is not None:
            features.append(self.ct_backbone(ct))
        fused = torch.stack(features).mean(dim=0)
        return self.classifier(fused)

# === Confident label ===
def get_confident_label(probs, threshold=0.6):
    confident = (probs >= threshold).any(dim=0).int()
    return confident

# === Training ===
def train_multimodal_model(mri_root, ct_root):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    dataset = MultiModalDataset(mri_root, ct_root, final_label_dict)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalEfficientNet(num_classes=len(LABELS)).to(device)

    # Pos weight for BCEWithLogitsLoss
    label_list = [torch.tensor(final_label_dict[pid]) for pid in dataset.patient_ids]
    label_tensor = torch.stack(label_list).float()
    pos = label_tensor.sum(dim=0)
    neg = label_tensor.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    fp_vakas = defaultdict(list)
    fn_vakas = defaultdict(list)

    for epoch in range(40):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for mri, ct, labels, _ in train_loader:
            mri = mri.to(device)
            ct = ct.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(mri, ct)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(output) > 0.5).int()
            correct += (preds == labels.int()).all(dim=1).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # === Validation ===
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for mri, ct, labels, pids in val_loader:
                mri = mri.to(device)
                ct = ct.to(device)
                labels = labels.to(device)

                outputs = model(mri, ct)
                probs = torch.sigmoid(outputs)

                batch_preds = []
                for i in range(probs.size(0)):
                    confident_pred = get_confident_label(probs[i].unsqueeze(0), threshold=0.6)
                    batch_preds.append(confident_pred)

                preds = torch.stack(batch_preds)
                all_preds.append(preds.cpu())
                all_labels.append(labels.int().cpu())
                val_correct += ((preds & labels.int()).sum(dim=1) > 0).sum().item()
                val_total += labels.size(0)

                # Collect FP/FN vaka IDs
                for i, pid in enumerate(pids):
                    for cls_idx, cls_name in enumerate(LABELS):
                        pred = preds[i, cls_idx].item()
                        true = labels[i, cls_idx].item()
                        if pred == 1 and true == 0:
                            fp_vakas[cls_name].append(pid)
                        elif pred == 0 and true == 1:
                            fn_vakas[cls_name].append(pid)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        TP = ((all_preds == 1) & (all_labels == 1)).sum(dim=0).float()
        FP = ((all_preds == 1) & (all_labels == 0)).sum(dim=0).float()
        FN = ((all_preds == 0) & (all_labels == 1)).sum(dim=0).float()

        epsilon = 1e-7
        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        f1_per_class = 2 * precision * recall / (precision + recall + epsilon)

        f1_macro = f1_per_class.mean().item()
        f1_micro = (2 * TP.sum()) / (2 * TP.sum() + FP.sum() + FN.sum() + epsilon)
        val_acc = (all_preds == all_labels).all(dim=1).float().mean().item()

        print(f"[MULTI] Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | F1 Macro: {f1_macro:.4f} | F1 Micro: {f1_micro:.4f}")

    # === Save model and print FP/FN after training ===
    torch.save(model.state_dict(), "resnet_multimodal.pth")
    print("✅ Model saved as resnet_multimodal.pth")

    print("\n🟥 False Positives (FP) — Vaka IDs by class:")
    for cls_name, pid_list in fp_vakas.items():
        print(f"  {cls_name}: {sorted(set(pid_list))}")

    print("\n🟦 False Negatives (FN) — Vaka IDs by class:")
    for cls_name, pid_list in fn_vakas.items():
        print(f"  {cls_name}: {sorted(set(pid_list))}")

# === Run ===
if __name__ == "__main__":
    mri_root = r"C:\Users\user\Desktop\weighted_brain_sequences_2"
    ct_root = r"C:\Users\user\Desktop\sliding_masked_Vakalar"
    train_multimodal_model(mri_root, ct_root)
