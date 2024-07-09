# UniVaR Training

Our training dataset for UniVar can be accessed from [Google Drive](https://drive.google.com/file/d/1bnBkGQz6EhJZfY4FLLmCgIHSvj_k_V_8/view?usp=sharing). The training of UniVar is done with a codebase modified from [`contrastors`](https://github.com/nomic-ai/contrastors) (cloned from [this commit](https://github.com/nomic-ai/contrastors/commit/20f395409bb759708c6c0310b9cd2ae91583db3d)). For details of our implementation, please refer to the following pieces of codes:

- Implementing a `MultiViewValueQADataset` class for data loading and view sampling: [code](https://github.com/ChenDelong1999/contrastor-value-embedding/blob/main/src/contrastors/dataset/torch_loader.py#L42)
- Registering additional arguments required by `MultiViewValueQADataset`: [code](https://github.com/ChenDelong1999/contrastor-value-embedding/blob/main/src/contrastors/config.py#L113-L116)
- Add `MultiViewValueQADataset` to `clip.py` to use the contractor implemented InfoNCE loss: [code](https://github.com/ChenDelong1999/contrastor-value-embedding/blob/main/src/contrastors/trainers/clip.py#L78)
- Prepare a training config `.yaml` file (modified from [contrastive_pretrain.yaml](https://github.com/nomic-ai/contrastors/blob/20f395409bb759708c6c0310b9cd2ae91583db3d/src/contrastors/configs/train/contrastive_pretrain.yaml) in contractor): [code](https://github.com/ChenDelong1999/contrastor-value-embedding/blob/main/src/contrastors/configs/train/value_embedding.yaml)
