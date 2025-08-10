# Symmetry_Aware_Brain_CT_Slide_Reader_AI_with_binary_output_for_stroke
Symmetry-Aware AI designed to produce binary outputs for brain strokes from single brain ct slides.

This code was initially made for the purpose of the national hackathon but we achieved the highest result anyone can achieve in Turkey. Yes like every good project that is made inside the borders of country Turkey  we were eliminated from the competition :) It was a great honor to achieve such results for us and now we would like to share the contributions with you so the things we discovered may help others.

# Medical Imaging Preprocessing and Sequence Classification Pipeline

This repository presents a complete methodology for preprocessing MRI and CT DICOM datasets, normalizing intensity values, removing noise, separating sequences, and implementing a hybrid expert–AI decision system for lesion detection.

---

## Converting MR and CT DICOM Data into Normalized Grayscale NumPy Arrays

When analyzing MR DICOM files, we observed significant variability in pixel values.  
For example, in some images, the cerebrospinal fluid (CSF) appeared around **500**, while in others it reached **1500**.  
However, relative intensity relationships remained consistent — in T2-weighted sequences, CSF always appeared hyperintense regardless of raw pixel value.

We leveraged this proportional consistency to convert the images into **normalized grayscale values**, enabling more robust downstream processing.

For CT DICOM files, we found that some scans were improperly windowed for brain parenchyma.  
To address this, we **standardized windowing parameters** across all CT images and reconstructed the data into NumPy arrays.

---

## Data Preprocessing and Sequence Separation

The provided DICOM data was **unorganized and lacked metadata**, meaning we could not directly identify which sequence each slice belonged to.  
Due to limited data, brute-force methods did not work well.

After manual inspection, we identified these sequences:
- **T2**
- **T2 FLAIR**
- **DWI b0**
- **DWI b1000**
- **ADC**

We manually separated them initially, but recognizing the inefficiency, we developed an **automatic sequence classification algorithm**.

---

## Removing Background Noise and Isolating Brain Parenchyma

Our NumPy arrays contained **background noise** outside the brain region.  
We applied an additional preprocessing step that **isolated only brain parenchyma** and set all other values to zero.  
The output was a clean NumPy array containing **only brain tissue**.

---

## Customized Sliding Blur and Hyper/Hypointensity Segmentation

From the brain parenchyma NumPy arrays, we:
1. Calculated **mean pixel intensity per sequence**.
2. Computed **z-scores** for each slice.
3. Mapped z-scores into discrete grayscale values between 0 and 255:

| Z-score Range      | Grayscale Value |
|--------------------|-----------------|
| < -2               | 0               |
| -2 to -1           | 51              |
| -1 to 0            | 102             |
| 0 to 1             | 153              |
| 1 to 2             | 204              |
| > 2                | 255              |

Background pixels (all four neighbors = 0) were excluded from calculations to avoid skewing results.

This approach visually emphasized lesions, making them stand out more clearly.  
Although this introduced **lossy quantization**, it significantly improved **explainability**.



---

## Sequence-Based 3D Analysis

In clinical practice, radiologists never diagnose from a single sequence alone — they compare the **same brain region across multiple sequences**.

To mimic this reasoning in AI, we stacked different sequences like **RGB channels** (e.g., acute, subacute, chronic) and developed a scoring system.  
The result was an **intuitively designed model architecture**.

This heuristic model:
- **Validates** CNN predictions
- **Flags cases** for expert review when model outputs disagree

By combining expert knowledge with AI decision-making, we built a **safety layer** to strengthen human–AI collaboration.

---

## Example
<img width="250" height="242" alt="resim_2025-08-10_124813891" src="https://github.com/user-attachments/assets/c19a8324-0ef9-4313-9614-5e601711563b" />
<img width="253" height="257" alt="resim_2025-08-10_124837323" src="https://github.com/user-attachments/assets/8ea1b009-a016-4c9c-bb75-42f67117ce0e" />



**Figure:** Diffusion-weighted imaging (DWI) and corresponding lesion segmentation mask.  
Left: Axial DWI slice showing a hyperintense ischemic lesion.  
Right: Segmentation mask highlighting the lesion area.

---

## Key Features
- **MR and CT normalization** for consistent pixel interpretation
- **Automated sequence separation** after manual verification
- **Background noise removal** for cleaner inputs
- **Quantized z-score mapping** for explainable lesion visualization
- **3D multi-sequence integration** inspired by radiology workflow
- **Expert–AI hybrid decision layer** for improved reliability

---

## License
This project is licensed under the GNU License.
