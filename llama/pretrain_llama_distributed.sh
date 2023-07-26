#! /bin/bash

DATA_PATH=$1
CHECKPOINT_PATH=$2
TOKENIZER_PATH=$3
TENSORBOARD_DIR=$4
TP_SIZE=$5
PP_SIZE=$6

NNODES=$7

GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1


# Runs the "13B" parameter model
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
"

LLAMA_ARGS="
    --tokenizer-model $TOKENIZER_PATH \
    --num-layers 40 \
    --hidden-size 5120 \
    --ffn-hidden-size 13824 \
    --num-attention-heads 40 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --swiglu \
    --layernorm-epsilon 1e-6 \
    --use-rotary-position-embeddings \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --weight-decay 0.1 \
    --micro-batch-size 2 \
    --global-batch-size 2000 \
    --train-iters 250000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --no-load-optim \
    --no-load-rng \
    --lr 0.000015 \
    --lr-decay-style cosine \
    --min-lr 0.000005 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .00 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --initial-loss-scale 16384.0 \
    --dataloader-type cyclic \
"

INFER_ARGS="
    --inference-batch-times-seqlen-threshold 32, \
    --max-tokens-to-oom 2048
"

LOG_ARGS="
    --distributed-timeout-minutes 120 \
    --timing-log-level 2 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-validation-ppl-to-tensorboard \
    --log-timers-to-tensorboard \
    --tensorboard-dir $TENSORBOARD_DIR
"

DATA_ARGS="
    --dataset-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2500 \
    --eval-interval 100 \
    --eval-iters 1 \
"

OPT_ARGS="
    --distributed-backend nccl \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --sequence-parallel \
    --recompute-activations \
    --use-cpu-initialization \
    --use-distributed-optimizer \
    --use-flash-attn \
    --no-bias-gelu-fusion \
    --fp16
"

# Run for 100 iterations and save checkpoint at 50
torchrun $DISTRIBUTED_ARGS \
       pretrain_llama.py \
       $LLAMA_ARGS \
       $LOG_ARGS \
       $DATA_ARGS \
       $OUTPUT_ARGS \
       $OPT_ARGS

echo 50 > $CHECKPOINT_PATH/latest_checkpointed_iteration.txt