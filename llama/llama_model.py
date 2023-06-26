# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Boss zhipin, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import torch
import torch.nn.functional as F

from apex.normalization import (
    FusedRMSNorm,
    MixedFusedRMSNorm
)

from megatron import get_args
from megatron.core import mpu, tensor_parallel
from megatron.model.enums import AttnMaskType, AttnType, LayerType, ModelType
from megatron.model.language_model import TransformerLanguageModel, parallel_lm_logits
from megatron.model.module import MegatronModule
from megatron.model.transformer import (
    ParallelAttention,
    ParallelMLP,
    NoopTransformerLayer,
    ParallelTransformerLayer,
    ParallelTransformer,
)
from megatron.model.utils import init_method_normal, scaled_init_method_normal


class RMSNorm(MixedFusedRMSNorm):
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 sequence_parallel=False):
        args = get_args()
        super(RMSNorm, self).__init__(normalized_shape=hidden_size, eps=eps)
        self.sequence_parallel = sequence_parallel
        if args.use_cpu_initialization:
            self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=args.params_dtype))
        else:
            self.weight = torch.nn.Parameter(
                torch.ones(hidden_size, device=torch.cuda.current_device(), dtype=args.params_dtype))
        if sequence_parallel:
            # set sequence parallelism flag on weight parameters
            setattr(self.weight, 'sequence_parallel', self.sequence_parallel)


class Attention(ParallelAttention):
    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.causal):
        super(Attention, self).__init__(init_method, output_layer_init_method,
                                        layer_number, attention_type, attn_mask_type)

class FeedForward(ParallelMLP):
    def __init__(self, init_method, output_layer_init_method):
        args = get_args()

        if args.multiple_of:
            tmp = 4 * args.hidden_size
            tmp = int(2 * tmp / 3)
            args.ffn_hidden_size = \
                args.multiple_of * ((tmp + args.multiple_of - 1) // args.multiple_of)
        
        super(FeedForward, self).__init__(init_method, output_layer_init_method)

        assert self.swiglu == True, ('Use silu according Meta LLAMA code.')


class TransformerBlock(ParallelTransformerLayer):
    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.causal,
                 drop_path_rate=0.):
        args = get_args()

        # LLAMA is a decoder only model, but in Megatron model building we need to regard it is encoder only.
        super(TransformerBlock, self).__init__(init_method=init_method, 
                                               output_layer_init_method=output_layer_init_method,
                                               layer_number=layer_number, layer_type=layer_type,
                                               self_attn_mask_type=self_attn_mask_type,
                                               drop_path_rate=drop_path_rate)

        # Layernorm on the input data(attention_norm).
        self.input_layernorm = RMSNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            sequence_parallel=args.sequence_parallel)

        # Self attention.
        self.self_attention = Attention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)

        # Layernorm on the attention output(ffn_norm).
        self.post_layernorm = RMSNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            sequence_parallel=args.sequence_parallel)

        # MLP
        self.mlp = FeedForward(init_method, output_layer_init_method)

        assert self.apply_residual_connection_post_layernorm == False


class Transformer(ParallelTransformer):
    """Transformer class."""
    def __init__(self, init_method, output_layer_init_method,
                 model_type=ModelType.encoder_or_decoder,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_layer_norm=True,
                 pre_process=True, post_process=True,
                 drop_path_rate=0.0):
        args = get_args()

        assert args.transformer_impl == 'local', \
            ('For now, we don\'t have Hopper GPU, and we need to use RMSNorm. \
             So Transformer Engine (TE) is not available.')
    
        super(Transformer, self).__init__(init_method=init_method, 
                                          output_layer_init_method=output_layer_init_method,
                                          model_type=model_type, layer_type=layer_type, 
                                          self_attn_mask_type=self_attn_mask_type, 
                                          post_layer_norm=post_layer_norm,
                                          pre_process=pre_process, 
                                          post_process=post_process, 
                                          drop_path_rate=drop_path_rate)

        assert self.retro_add_retriever == False, ('No retro for now.')
        self.post_layer_norm == True

        # Transformer layers.
        def build_layer(layer_number):
            return TransformerBlock(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1])

        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            assert args.model_type != ModelType.encoder_and_decoder
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                args.num_layers // args.virtual_pipeline_model_parallel_size) + \
                (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if args.model_type == ModelType.encoder_and_decoder and \
                    mpu.get_pipeline_model_parallel_world_size() > 1:
                pipeline_rank = mpu.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

        if self.num_layers == 0:
            # When a standalone embedding stage is used (e.g.,
            # args.standalone_embedding_stage == True), virtual pipeline ranks
            # on pipeline rank 0 will have zero transformer layers assigned to
            # them. This results in the model's input and output tensors to be
            # the same, which will cause failure for certain output tensor
            # optimizations (e.g., pipeline output deallocation). To remedy
            # this, we assign a 'no-op' layer on these ranks, which will
            # disconnect the input tensor from the output tensor.
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([ NoopTransformerLayer(1) ])
        else:
            self.layers = torch.nn.ModuleList(
                [build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = RMSNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                sequence_parallel=args.sequence_parallel)


class LlamaLanguageModel(TransformerLanguageModel):
    def __init__(self,
                 init_method,
                 output_layer_init_method,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 add_encoder=True,
                 add_decoder=False,
                 decoder_attn_mask_type=AttnMaskType.causal,
                 add_pooler=False,
                 pre_process=True,
                 post_process=True):
        # LLAMA is a decoder only model, but in Megatron model building we need to regard it is encoder only.
        super(LlamaLanguageModel, self).__init__(init_method=init_method, 
                                                 output_layer_init_method=output_layer_init_method,
                                                 encoder_attn_mask_type=encoder_attn_mask_type, 
                                                 num_tokentypes=num_tokentypes,
                                                 add_encoder=add_encoder, add_decoder=add_decoder, 
                                                 decoder_attn_mask_type=decoder_attn_mask_type,
                                                 add_pooler=add_pooler, 
                                                 pre_process=pre_process, post_process=post_process)

        assert self.use_rotary_position_embeddings == True, ('Use rotary embedding according to Meta code.')
        assert self.add_decoder == False, \
            ('LLAMA is a decoder only model, but in Megatron model building we need to regard it is encoder only.') 
        # assert self.pre_process == True, ('Use Embedding')
        # assert self.post_process == True, (' a norm layer and a linear layer will be applied to the output.')
        assert self.untie_embeddings_and_output_weights == True, ('Use a ouput linear layer for computing last logits.')

        self.encoder = Transformer(
            init_method=self.init_method,
            output_layer_init_method=output_layer_init_method,
            self_attn_mask_type=self.encoder_attn_mask_type,
            post_layer_norm=True,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

def get_language_model(num_tokentypes, add_pooler,
                       encoder_attn_mask_type, init_method=None,
                       scaled_init_method=None, add_encoder=True,
                       add_decoder=False,
                       decoder_attn_mask_type=AttnMaskType.causal,
                       pre_process=True, post_process=True):
    """Build language model and return along with the key to save."""
    args = get_args()

    if init_method is None:
        init_method = init_method_normal(args.init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

    # Language model.
    language_model = LlamaLanguageModel(
        init_method,
        scaled_init_method,
        encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process
    )
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key

def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy,
                                   return_logits=False):

    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if labels is None:
        # gather along the vocab dimension
        output = mpu.gather_from_tensor_model_parallel_region(output)
        # [s b h] => [b s h]
        return output.float().transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0,1).contiguous()
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)
        
        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        if return_logits:
            # gather along the vocab dimension
            output = mpu.gather_from_tensor_model_parallel_region(output)
            return loss, output.float().transpose(0,1).contiguous()
        else:
            return loss

class LLAMAModel(MegatronModule):
    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 return_logits=False):
        args = get_args()
        super(LLAMAModel, self).__init__(share_word_embeddings=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        self.return_logits = return_logits

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers),
            pre_process=self.pre_process,
            post_process=self.post_process)
        
        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings(init_method_normal)
    
    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None,
                encoder_input=None):

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params,
            encoder_input=encoder_input)

        if self.post_process:
            return post_language_model_processing(
                lm_output, labels,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.word_embeddings_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy,
                return_logits=self.return_logits)
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
