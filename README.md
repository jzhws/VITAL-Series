# VITAL-Series
Official implementation of VITAL: Vision-Encoder centered pretraining for LMMs in visual quality assessment.

# VITAL Model Usage Guide

## 📥 Model Download

1. **Download the models**:
   - **VITAL-Assistant-8B**, **VITAL-Base-8B**, and **VITAL-internvit-SF-400M** (Visual Encoder) should be placed under the `VITAL` folder.
   - Additional zero or warm-up series models can be downloaded from HuggingFace.
   
2. **For testing the extended functionality of the VITAL Visual Encoder** (e.g., lightweight linear-probe), place **VITAL-internvit-SF-400M** under the `VITAL_linear_probe` folder.

## ⚙️ VITAL Main Model (Using Large Model Architecture)

### 🧪 Testing

1. **Modify the JSON files** located in `shell/eval/eval_data` (there are two JSON files, one for scoring and one for description tasks):
   - Update the `root` and `annotation` parameters with the image/video directory to be tested and the corresponding JSON files (example JSON files for scoring and description tasks are in `/shell/eval/custom`).
   
2. **Run batch tests**:
   - Use the following scripts for batch testing the scoring and description tasks:
     - `evaluate_custom_scoring.sh` for the scoring task.
     - `evaluate_custom_description.sh` for the description task.

3. **Scoring and Description Task Files**:
   - The specific Python files to run the scoring/description tasks are located in `internvl/eval`.
   - By default, scoring uses `scoring.py`. If you need to score videos quickly, use `scoring_less_token.py`.
     - To switch to this file, modify line 31 in `evaluate_custom_scoring.sh` to point to this script.

### 🏋️‍♂️ Training

1. Use the shell scripts from `training_shell` to begin training (modify the file paths as instructed):
   - `pretrain.sh` for pre-training.
   - `warm_up.sh` for quick adaptation training.

---

## 👁️‍🗨️ VITAL Linear Probe (Visual Encoder Extension)

### VITAL-Vision-Encoder Usage Guide

This section covers the training/testing methods for adding non-LLM structures (e.g., linear probes) to **VITAL-Vision-Encoder**.

#### 🏋️‍♂️ Training

1. Run `shell/probe_finetune.sh` (modify the file paths as instructed in the script).

#### 🧪 Testing

1. Run `shell/evaluate_video.sh` (adjust the file paths based on the instructions provided).

---
## Model Zoo

### 📦 VITAL Models

- **VITAL-Base-8B** (Use the code in the `VITAL-LMM` folder for training/testing):  
  [JZHWS/VITAL-Base-8B at main](https://huggingface.co/JZHWS/VITAL-Base-8B/tree/main)
  
- **VITAL-Assistant-8B** (Use the code in the `VITAL-LMM` folder for training/testing):  
  [JZHWS/VITAL-Assistant-8B · Hugging Face](https://huggingface.co/JZHWS/VITAL-Assistant-8B)

- **VITAL-Warm-up-1B** (Use the code in the `VITAL-LMM` folder for training/testing):  
  [JZHWS/VITAL-Warm-up-1B · Hugging Face](https://huggingface.co/JZHWS/VITAL-Warm-up-1B)

- **VITAL-Warm-up-2B** (Use the code in the `VITAL-LMM` folder for training/testing):  
  [JZHWS/VITAL-Warm-up-2B · Hugging Face](https://huggingface.co/JZHWS/VITAL-Warm-up-2B)

- **VITAL-Warm-up-14B** (Use the code in the `VITAL-LMM` folder for training/testing):  
  [JZHWS/VITAL-Warm-up-14B · Hugging Face](https://huggingface.co/JZHWS/VITAL-Warm-up-14B)

### 👁️ VITAL Vision Encoder & Extensions

- **VITAL-Vision-Encoder-300M** (Use the code in the `VITAL-linear-probe` folder, suitable for downstream tasks fine-tuning or model structure transfer, see Linear-probe usage and VITAL-Zero model construction):  
  [JZHWS/VITAL-Vision-Encoder-300M · Hugging Face](https://huggingface.co/JZHWS/VITAL-Vision-Encoder-300M)

- **VITAL-Linear-Probe** (Use the code in the `VITAL-linear-probe` folder for training/testing):  
  [JZHWS/VITAL-Linear-Probe · Hugging Face](https://huggingface.co/JZHWS/VITAL-Linear-Probe)

Feel free to adjust the file paths and parameters as necessary according to your setup. For any other questions or issues, please refer to the official documentation or open an issue on this repository!
