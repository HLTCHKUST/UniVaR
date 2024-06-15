# UniVaR Training

Our training dataset for UniVar can be accessed from this [link](https://drive.google.com/file/d/1bnBkGQz6EhJZfY4FLLmCgIHSvj_k_V_8/view?usp=sharing).

The training of UniVar is done with the [`contrastors`](https://github.com/nomic-ai/contrastors) codebase (cloned from [this commit](https://github.com/nomic-ai/contrastors/commit/20f395409bb759708c6c0310b9cd2ae91583db3d)). Some of the key hyperparameters used in our training are as follows:

```yaml

Trainig Arguments:

  num_epochs: 3 
  batch_size: 64 # we use 4 GPU for training, leading to an effective batch size of 256
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_steps: 700
  schedule_type: "inverse_sqrt"
  max_grad_norm: 1.0
  adam_beta1: 0.9
  adam_beta2: 0.999
  loss_fn: "clip"

Model Arguments:

  logit_scale: 50
  trainable_logit_scale: false
  model_type: "encoder"
  seq_len: 1024 # for lambda=5 QA per view
  rotary_emb_fraction: 0.0
  pad_vocab_to_multiple_of: 64 
  use_rms_norm: false
  activation_function: "gelu"
  pooling: "mean"
  tokenizer_name: "bert-base-uncased"
  model_name: "nomic-ai/nomic-embed-text-v1" # initialization of UniVar models

```

