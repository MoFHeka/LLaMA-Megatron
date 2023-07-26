"""Pretrain LLAMA"""

import os

import torch
from functools import partial
from megatron import get_args
from megatron.arguments import core_transformer_config_from_args
from megatron import print_rank_0
from megatron import get_timers
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from custom_training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

# from llama_model import LLAMAModel
from llama_model_debug import LLAMAModel

from megatron.core import tensor_parallel


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
    """Generate a batch"""
    args = get_args()
    #tokenizer = get_tokenizer()

    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    #print("tokens:\t", data_b['text'], "\ttoken_size\t", data_b['text'].size())
    tokens_ = data_b['text'].long().cuda()
    labels = tokens_[:, 1:].cuda().contiguous()
    tokens = tokens_[:, :-1].cuda().contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        0,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    #print("tokens:\t", tokens)
    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


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
        'multiple_of': 256,  # make SwiGLU hidden layer size multiple of large power of 2
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
        'recompute_granularity': 'selective',
        # activation checkpointing (or gradient checkpointing) for reducing activation recomputation. 'selective' or 'full'.
    }

    pretrain(None, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=_add_llama_model_args,
             args_defaults=model_args_defaults
             )