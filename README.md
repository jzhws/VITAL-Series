# VITAL-Series
Official implementation of VITAL: Vision-Encoder centered pretraining for LMMs in visual quality assessment.
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="teaser.png">
</div>
## ⚙️ Environment Setup

To install the necessary environment, please use the provided `environment.yml` file. It contains all the dependencies required for running the models.


## 📥 Model Download

1. **Download the models**:
   - **VITAL-Assistant-8B**, **VITAL-Base-8B**, and **VITAL-Vision-Encoder-300M** (Visual Encoder) should be placed under the `VITAL-LMM` folder.
   - Additional zero or warm-up series models can be downloaded from HuggingFace.
   
2. **For testing the extended functionality of the VITAL Visual Encoder** (e.g., lightweight linear-probe), place **VITAL-Vision-Encoder-300M** under the `VITAL-linear-probe` folder.

## ⚙️ VITAL Main Models (Using LMM Architecture)
`cd VITAL-LMM`

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
`cd VITAL-linear-probe`
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
## 📚 Citation

When using the related models, please kindly cite the following reference articles:

```bibtex
@article{jia2025vital,
  title={VITAL: Vision-Encoder-centered Pre-training for LMMs in Visual Quality Assessment},
  author={Jia, Ziheng and Cao, Linhan and Han, Jinliang and Zhang, Zicheng and Qian, Jiaying and Wang, Jiarui and Chen, Zijian and Zhai, Guangtao and Min, Xiongkuo},
  journal={arXiv preprint arXiv:2511.17962},
  year={2025}
}

@inproceedings{jia2025vqa2,
  title={Vqa2: visual question answering for video quality assessment},
  author={Jia, Ziheng and Zhang, Zicheng and Qian, Jiaying and Wu, Haoning and Sun, Wei and Li, Chunyi and Liu, Xiaohong and Lin, Weisi and Zhai, Guangtao and Min, Xiongkuo},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={6751--6760},
  year={2025}
}

@inproceedings{zhang2025q,
  title={Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs},
  author={Zhang, Zicheng and Jia, Ziheng and Wu, Haoning and Li, Chunyi and Chen, Zijian and Zhou, Yingjie and Sun, Wei and Liu, Xiaohong and Min, Xiongkuo and Lin, Weisi and others},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={3229--3239},
  year={2025}
}


Feel free to adjust the file paths and parameters as necessary according to your setup. For any other questions or issues, please refer to the official documentation or open an issue on this repository!
