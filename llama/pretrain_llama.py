"""Pretrain LLAMA"""

import os
import torch
from torch import Tensor
from functools import partial
from typing import Union
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.training import pretrain
from megatron.utils import (
    get_ltor_masks_and_position_ids,
    get_batch_on_this_cp_rank,
    average_losses_across_data_parallel_group
)
from megatron.arguments import core_transformer_config_from_args

from llama_model import LLAMAModel


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    args = get_args()
    config = core_transformer_config_from_args(args)

    print_rank_0('building LLAMA model ...')
    model = LLAMAModel(
        config=config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        return_logits=False
    )
    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()

def loss_func(loss_mask: Tensor, output_tensor: Tensor):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat(
            [torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)]
        )
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (LlamaModel): The LLaMa Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def dataset_preprocess(dataset_path):
    # The data prefix should be in the format of:
    #   weight-1, data-prefix-1, weight-2, data-prefix-2, ..
    datasets_folder = ['data-1', 'data-2']
    datasets_weight = [40., 60.]
    datafile_suffix = '.bin'
    data_prefix = []

    if torch.distributed.is_initialized():
        for ds_w, ds_f in zip(datasets_weight, datasets_folder):
            ds_dir = os.path.join(dataset_path, ds_f)
            all_files = os.listdir(ds_dir)
            ds_w_one = ds_w / len(all_files)
            for file in all_files:
                if file.endswith(datafile_suffix):
                    data_prefix.append(ds_w_one)
                    data_prefix.append(os.path.join(ds_dir, file.split(datafile_suffix)[0]))

    return data_prefix


def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    if args.dataset_path:
        args.data_path = dataset_preprocess(args.dataset_path)
    print_rank_0('> rank 0 training data files are:')
    print_rank_0(args.data_path)

    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        blend_per_split=[
            args.train_data_path,
            args.valid_data_path,
            args.test_data_path,
        ],
        split=args.split,
        path_to_cache=args.data_cache_path
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for LLAMA ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset, train_val_test_num_samples, core_gpt_dataset_config_from_args(args)
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def _add_llama_model_args(parser):
    group = parser.add_argument_group(title='LLAMA-Model-Args')
    group.add_argument('--multiple-of', type=int, default=None,
                       help='Make SwiGLU hidden layer size multiple of large power of 2.'),
    group.add_argument('--dataset-path', type=str, default=None,
                       help='A path for custom dataset preprocess.'),
    return parser


if __name__ == "__main__":    
    
    model_args_defaults = {
        # model options
        'tokenizer_type': 'SentencePieceTokenizer',
        # 'num_layers': 8,
        # 'hidden_size': 512,
        # 'num_attention_heads': 8,
        # 'untie_embeddings_and_output_weights': True,
        'ffn_hidden_size': None,
        'multiple_of': 256, # make SwiGLU hidden layer size multiple of large power of 2
        'swiglu': True,
        # 'layernorm_epsilon': 1e-5, 
        # 'add_bias_linear': False,
        # 'use_rotary_position_embeddings': True,
        # 'attention_dropout': 0.0,
        # inference options
        # 'inference_batch_times_seqlen_threshold': 32,
        # 'max_tokens_to_oom': 2048,
        # optimization options
        # 'use_flash_attn': True,
        'use_cpu_initialization': True,
        # 'perform_initialization': True,
        # 'gradient_accumulation_fusion': True,
        # 'sequence_parallel': False,
        # 'async_tensor_model_parallel_allreduce': True,
        'recompute_granularity': 'selective', # activation checkpointing (or gradient checkpointing) for reducing activation recomputation. 'selective' or 'full'.
    }

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=_add_llama_model_args,
             args_defaults=model_args_defaults
    )