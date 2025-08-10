import os
import json
import numpy as np
from tqdm import tqdm

# === PARAMETERS ===
input_root = r"C:\Users\user\Desktop\only_brain_sequences"
output_root = r"C:\Users\user\Desktop\sliding_masked_Vakalar_2"
lesion_json_path = r"C:\Users\user\Desktop\TEKNOFEST_PUBLIC\MR_Yeni (1) (1).json"
window_size = 4
half = window_size // 2

# Create output folder
os.makedirs(output_root, exist_ok=True)

# === Load JSON: mapping ImageId -> LessionTypeName ===
with open(lesion_json_path, 'r', encoding='utf-8') as f:
    lesion_data = json.load(f)

# Build a dict like: { '50152720.0.167' : 'HiperakutAkut' }
image_lesion_map = {
    entry['ImageId'].replace('.dcm', ''): entry['LessionTypeName']
    for entry in lesion_data
}

# === STEP 1: Compute global mean/std from "HiperakutAkut" ===
print("🔍 Global mean/std hesaplanıyor... (sadece HiperakutAkut vakalardan)")
global_valid_means = []

for root, dirs, files in os.walk(input_root):
    for file in files:
        if not file.endswith(".npy"):
            continue

        # Strip 'only_brain_' and extension to get ImageId
        file_id = os.path.basename(file).replace('only_brain_', '').replace('.npy', '')
        lesion_type = image_lesion_map.get(file_id)

        if lesion_type != "HiperakutAkut":
            continue

        img_path = os.path.join(root, file)
        img = np.load(img_path)
        height, width = img.shape

        for i in range(half, height - half):
            for j in range(half, width - half):
                patch = img[i - half:i + half + 1, j - half:j + half + 1]
                if np.any(patch == 0):
                    continue
                global_valid_means.append(np.mean(patch))

if not global_valid_means:
    raise ValueError("❌ HiperakutAkut vakalardan geçerli patch bulunamadı.")

global_mean = np.mean(global_valid_means)
global_std = np.std(global_valid_means)
print(f"✅ Global mean: {global_mean:.4f}, std: {global_std:.4f}")

# === STEP 2: Image classification function using GLOBAL stats ===
def process_image(npy_path, global_mean, global_std):
    img = np.load(npy_path)
    height, width = img.shape
    output_img = np.zeros((height, width), dtype=np.uint8)

    for i in range(half, height - half):
        for j in range(half, width - half):
            patch = img[i - half:i + half + 1, j - half:j + half + 1]
            if np.any(patch == 0):
                continue
            mean_val = np.mean(patch)
            z = (mean_val - global_mean) / global_std

            # Map z-score to grayscale (6 classes)
            if z < -2:
                output_img[i, j] = 0
            elif -2 <= z < -1:
                output_img[i, j] = 51
            elif -1 <= z < 0:
                output_img[i, j] = 102
            elif 0 <= z < 1:
                output_img[i, j] = 153
            elif 1 <= z < 2:
                output_img[i, j] = 204
            else:
                output_img[i, j] = 255

    return output_img

# === STEP 3: Process all .npy files ===
print("🚀 Tüm görüntüler işleniyor...")

for root, dirs, files in os.walk(input_root):
    for file in files:
        if not file.endswith(".npy"):
            continue

        input_path = os.path.join(root, file)
        relative_path = os.path.relpath(input_path, input_root)
        output_dir = os.path.join(output_root, os.path.dirname(relative_path))
        os.makedirs(output_dir, exist_ok=True)

        print(f"⚙️ İşleniyor: {input_path}")
        try:
            result_img = process_image(input_path, global_mean, global_std)
            output_file = os.path.join(output_dir, file.replace(".npy", "_classified.npy"))
            np.save(output_file, result_img)
        except Exception as e:
            print(f"❌ Hata oluştu ({input_path}): {e}")

print("✅ Tamamlandı. Çıktılar:", output_root)
