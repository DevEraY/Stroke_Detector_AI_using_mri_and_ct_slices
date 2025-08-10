import os
import numpy as np
from glob import glob
import re

# Define intensity ranges and stage table
stage_table = {
    "HiperakutAkut": {
        "T2": "Hyperintense",
        "T2 FLAIR": "Hyperintense",
        "DWI b0": "Hypointense",
        "DWI b1000": "Hyperintense",
        "ADC": "Hypointense"
    },
    "Subakut": {
        "T2": "Hyperintense",
        "T2 FLAIR": "Hyperintense",
        "DWI b1000": "Hyperintense",
        "ADC": "Normal",
        "DWI b0": "Normal"
    },
    "NormalKronik": {
        "T2": "Hyperintense",
        "T2 FLAIR": "Hyperintense",
        "DWI b0": "Isointense",
        "DWI b1000": "Isointense",
        "ADC": "Hyperintense"
    }
}
stages = list(stage_table.keys())

# Base directories
base_dir = r"C:\Users\user\Desktop\sliding_masked_Vakalar_2"
output_base = r"C:\Users\user\Desktop\weighted_brain_sequences_2"
os.makedirs(output_base, exist_ok=True)

# Sequence folders to process
seq_folders = ['T2', 'T2 FLAIR', 'ADC', 'DWI b0', 'DWI b1000']

# Regex for slide numbers
slide_pattern = re.compile(r'.*\.(\d+)_classified\.npy$')

for vaka_folder in os.listdir(base_dir):
    if not vaka_folder.startswith("Vaka_"):
        continue
        
    vaka_path = os.path.join(base_dir, vaka_folder)
    print(f"Processing {vaka_folder}...")
    
    sequence_files = {seq: {} for seq in seq_folders}
    all_slide_nums = set()

    # Collect files
    for seq in seq_folders:
        possible_paths = [
            os.path.join(vaka_path, seq),
            os.path.join(vaka_path, "MRI", seq),
            os.path.join(vaka_path, "CT", seq)
        ]
        
        seq_path = None
        for path in possible_paths:
            if os.path.exists(path):
                seq_path = path
                break
        
        if seq_path is None:
            print(f"  Sequence folder not found: {seq}")
            continue
            
        print(f"  Found {seq} at {seq_path}")
        
        files = glob(os.path.join(seq_path, "only_brain_*.npy"))
        if not files:
            print(f"  No .npy files found in {seq}")
            continue
            
        for f in files:
            filename = os.path.basename(f)
            match = slide_pattern.match(filename)
            if match:
                slide_num = int(match.group(1))
                sequence_files[seq][slide_num] = f
                all_slide_nums.add(slide_num)
            else:
                print(f"  Couldn't parse slide number from: {filename}")
    
    # Sort slides
    sorted_sequences = {}
    for seq in seq_folders:
        if sequence_files[seq]:
            sorted_slides = sorted(sequence_files[seq].keys())
            sorted_sequences[seq] = [sequence_files[seq][s] for s in sorted_slides]
    
    max_slides = max(len(sorted_sequences[seq]) for seq in sorted_sequences) if sorted_sequences else 0
    if max_slides == 0:
        print(f"  No valid MRI files found in {vaka_folder}")
        continue

    vaka_out_dir = os.path.join(output_base, vaka_folder)
    os.makedirs(vaka_out_dir, exist_ok=True)
    print(f"  Output directory: {vaka_out_dir}")
    print(f"  Found {max_slides} slide groups to process")

    for group_idx in range(max_slides):
        group_arrays = {}
        group_shapes = set()
        
        for seq in seq_folders:
            if seq in sorted_sequences and group_idx < len(sorted_sequences[seq]):
                file_path = sorted_sequences[seq][group_idx]
                try:
                    arr = np.load(file_path)
                    group_arrays[seq] = arr
                    group_shapes.add(arr.shape)
                except Exception as e:
                    print(f"    Error loading {file_path}: {str(e)}")
        
        if not group_arrays:
            print(f"    No arrays for group {group_idx}")
            continue
            
        if len(group_shapes) > 1:
            print(f"    Inconsistent shapes for group {group_idx}: {group_shapes}")
            continue
            
        height, width = next(iter(group_shapes))
        points_arr = np.zeros((height, width, 3), dtype=np.float32)
        all_zero_mask = np.ones((height, width), dtype=bool)

        # === Store intensity masks for logic-based combination
        masks = {}
        for seq, arr in group_arrays.items():
            masks[seq] = {
                "hyper": (arr >= 154) & (arr <= 255),
                "hypo": (arr >= 1) & (arr <= 102),
                "iso": (arr >= 103) & (arr <= 153),
                "raw": arr
            }
            all_zero_mask &= (arr == 0)

        # === Custom override logic for acute stage
        if ("DWI b1000" in masks and "DWI b0" in masks):
            acute_mask = masks["DWI b1000"]["hyper"] & masks["DWI b0"]["hypo"]
            points_arr[..., 0][acute_mask] = 100  # HiperakutAkut

        if ("T2" in masks and "T2 FLAIR" in masks):
            acute_mask_2 = masks["T2"]["hyper"] & masks["T2 FLAIR"]["hyper"]
            points_arr[..., 0][acute_mask_2] = 100  # HiperakutAkut

        # === Normal point addition logic
        for seq, arr in group_arrays.items():
            for stage_idx, stage in enumerate(stages):
                expected = stage_table[stage].get(seq)
                weight = 5  # default weight

                if seq == "DWI b1000" and expected == "Hyperintense":
                    weight = 10
                elif seq == "DWI b0" and expected == "Hypointense":
                    weight = 10
                elif seq == "T2 FLAIR" and expected == "Hyperintense":
                    weight = 10
                elif seq == "T2" and expected == "Hyperintense":
                    weight = 10
                elif seq == "ADC" and expected in ["Hyperintense", "Hypointense"]:
                    weight = 10

                if expected == "Hyperintense":
                    points_arr[..., stage_idx] += masks[seq]["hyper"] * weight
                elif expected == "Hypointense":
                    points_arr[..., stage_idx] += masks[seq]["hypo"] * weight
                elif expected in ("Isointense", "Normal"):
                    points_arr[..., stage_idx] += masks[seq]["iso"] * weight

        points_arr /= len(group_arrays)

        for channel in range(3):
            channel_data = points_arr[..., channel]
            channel_data[all_zero_mask] = 0

        sample_file = sorted_sequences[next(iter(group_arrays))][group_idx]
        fname = os.path.basename(sample_file).replace('_classified', '_weighted')
        output_path = os.path.join(vaka_out_dir, fname)

        np.save(output_path, points_arr)
        print(f"    Saved group {group_idx} as {fname}")


print("Processing complete!")