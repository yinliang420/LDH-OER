# Decoding Active Sites in High-Entropy Catalysts via Attention-Enhanced Model

This repository contains the code, data, and trained models for our paper published in *Science Advances*:

> **L. Yin, T. Ma, Z. Zhu, Z. Yao, S. Yu, Y. Li, C. Li, N. Ran, W. Zhou, J. Liu**, "Decoding active sites in high-entropy catalysts via attention-enhanced model," *Science Advances* **12**, eaea1170 (2026). DOI: [10.1126/sciadv.aea1170](https://doi.org/10.1126/sciadv.aea1170)

## Data & Checkpoints

All datasets and model checkpoints are hosted on Hugging Face:

🔗 **[https://huggingface.co/datasets/yinliang22/oer_dataset](https://huggingface.co/datasets/yinliang22/oer_dataset)**

Includes:
- Processed input data
- Trained model checkpoints
- Prediction dataset

Download via the `huggingface_hub` Python library:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yinliang22/oer_dataset",
    local_dir="./oer_dataset",
    repo_type="dataset"
)
```

## Citation

If you use this work, please cite:

```bibtex
@article{yin2026decoding,
  title   = {Decoding active sites in high-entropy catalysts via attention-enhanced model},
  author  = {Yin, Liang and Ma, Tiantian and Zhu, Zibo and Yao, Zhanao and Yu, Songlin and Li, Yi and Li, Chengbo and Ran, Nian and Zhou, Wei and Liu, Jianjun},
  journal = {Science Advances},
  volume  = {12},
  number  = {7},
  pages   = {eaea1170},
  year    = {2026},
  doi     = {10.1126/sciadv.aea1170}
}
```
