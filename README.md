# Decoding Active Sites in High-Entropy Catalysts via Attention-Enhanced Model

All datasets and model checkpoints associated with this work are publicly available on Hugging Face:

🔗 **[https://huggingface.co/datasets/yinliang22/oer_dataset](https://huggingface.co/datasets/yinliang22/oer_dataset)**

This repository includes:
- Processed input data
- Trained model checkpoints
- Prediction dataset

To download the data, simply visit the link above or use the `huggingface_hub` library in Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yinliang22/oer_dataset",
    local_dir="./oer_dataset",
    repo_type="dataset"
)

