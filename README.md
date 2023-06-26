# LLaMA-Megatron
A LLaMA Megatron implement.

# LLaMA 

This repository is intended as a minimal, hackable and readable example with Nivida [Megatron-LM](https://github.com/huggingface/Megatron-LM/tree/main) 
to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference.
In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5).

## Setup

In a conda env with pytorch / cuda available, run:
```bash
pip install -r requirements.txt

# Install Nvidia APEX
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

# Installing Megatron-LM
pip install git+https://github.com/MoFHeka/Megatron-LM.git
```
Then in this repository:
```bash
pip install -e .
```

## Download

Once your request is approved, you will receive links to download the tokenizer and model files.
Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

## Model

LLaMA modeling code was rebuilt on the basis of Megatron, showing in `llama_model.py`. Class LLAMAModel is the entry class.

## Checkpoint Transform

`tools/transform_huggingface_to_megatron.py` and `tools/transform_huggingface_to_megatron.py` was provided for converting llama model ckpt between Huggingface and Megatron.

## Pretrain

Firstly, we need to run `tools/preprocess_data.py` to generate the Megatron style pretrain text dataset. Or we could write our own pretrain code like `custom_pretrain_llama.py` with `custom_training.py`.

The provided `pretrain_llama.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts. Using `pretrain_llama_distributed.sh` to run it:
```bash
sh pretrain_llama_distributed.sh {dataset_folder} {ckpt_folder} {tokenizer_model} {tensorboard_folder} {tensor_parallel_size} {pipeline_parallel_size} {number_of_nodes}
```

Different models require different TP values:

|  Model | TP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 33B    | 4  |
| 65B    | 8  |

## Reference

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

## License
See the [LICENSE](LICENSE) file.
