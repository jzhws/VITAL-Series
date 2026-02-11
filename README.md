# VITAL-Series

Official implementation of **VITAL: Vision-Encoder-centered pretraining for LMMs in visual quality assessment**.

<p align="center">
  <img src="teaser.jpg" alt="VITAL teaser" width="100%" />
</p>

---

## ✨ Overview

VITAL-Series contains two major components:

- **VITAL-LMM**: training/evaluation code for VITAL main models.
- **VITAL-linear-probe**: visual encoder extension workflows (e.g., linear-probe and lightweight downstream adaptation).

---

## ⚙️ Environment Setup

Use the provided environment file:

```bash
conda env create -f environment.yml
```

If needed, adjust CUDA/PyTorch versions according to your machine.

---

## 📥 Model Download & Placement

1. Download **VITAL-Assistant-8B**, **VITAL-Base-8B**, and **VITAL-Vision-Encoder-300M**.
2. Place LMM-related models under `VITAL-LMM`.
3. For visual-encoder extension experiments (e.g., linear-probe), place **VITAL-Vision-Encoder-300M** under `VITAL-linear-probe`.
4. Additional zero/warm-up series models are available on Hugging Face (see [Model Zoo](#-model-zoo)).

---

## 🚀 VITAL Main Models (LMM)

```bash
cd VITAL-LMM
```

### 🧪 Testing

1. Edit JSON configs in `shell/eval/eval_data`:
   - Update `root` and `annotation` to your image/video paths and annotation files.
   - Example files are provided in `shell/eval/custom`.

2. Run batch evaluation scripts:

```bash
bash shell/eval/evaluate_custom_scoring.sh
bash shell/eval/evaluate_custom_description.sh
```

> Legacy wrappers `evaluate_custom_打分.sh` and `evaluate_custom_描述.sh` are kept for compatibility.

3. Evaluation entry scripts are in `internvl/eval`:
   - Default scoring: `scoring.py`
   - Faster video scoring: `scoring_less_token.py`

If you want to use `scoring_less_token.py`, modify line 31 in `shell/eval/evaluate_custom_scoring.sh` accordingly.

### 🏋️ Training

Use scripts in `training_shell` (update data/model paths before running):

```bash
bash training_shell/pretrain.sh
bash training_shell/warm_up.sh
```

---

## 👁️ VITAL Linear Probe (Visual Encoder Extension)

```bash
cd VITAL-linear-probe
```

This module supports training/testing with non-LLM heads (e.g., linear probes) on top of **VITAL-Vision-Encoder**.

### 🏋️ Training

```bash
bash shell/probe_finetune.sh
```

### 🧪 Testing

```bash
bash shell/evaluate_video.sh
```

Please update file paths in scripts for your local setup.

---

## 📦 Model Zoo

### VITAL Main Models

- **VITAL-Base-8B**: <https://huggingface.co/JZHWS/VITAL-Base-8B/tree/main>
- **VITAL-Assistant-8B**: <https://huggingface.co/JZHWS/VITAL-Assistant-8B>
- **VITAL-Warm-up-1B**: <https://huggingface.co/JZHWS/VITAL-Warm-up-1B>
- **VITAL-Warm-up-2B**: <https://huggingface.co/JZHWS/VITAL-Warm-up-2B>
- **VITAL-Warm-up-14B**: <https://huggingface.co/JZHWS/VITAL-Warm-up-14B>

### Vision Encoder & Extensions

- **VITAL-Vision-Encoder-300M**: <https://huggingface.co/JZHWS/VITAL-Vision-Encoder-300M>
- **VITAL-Linear-Probe**: <https://huggingface.co/JZHWS/VITAL-Linear-Probe>

---

## 📈 GitHub Daily Star/Issue Report

Generate a daily report for `jzhws1/VITAL-Series`:

```bash
python3 scripts/github_daily_report.py --repo jzhws1/VITAL-Series
```

Outputs:

- `reports/github_daily_report.md`: current stars, open issue count, and 24h issue updates.
- `reports/.github_daily_state.json`: previous snapshot used for delta calculation.

For scheduled runs:

```bash
cat scripts/github_daily_cron.example
```

Optional: set `GITHUB_TOKEN` to increase GitHub API rate limits.

---

## 📚 Citation

If you use this project, please cite:

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
```

---

For custom environments, adjust file paths and parameters as needed. If you encounter issues, feel free to open an issue in this repository.
