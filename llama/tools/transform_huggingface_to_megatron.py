import os

import torch

HUG_LOAD_PATH = "HUG_LOAD_PATH"
MEGATRON_SAVE_PATH = "MEGATRON_SAVE_PATH"

PP_SIZE = 1
TP_SIZE = 2
LAYER_NUM = 32
HEAD_NUM = 32
HIDDEN_DIM = 4096


def tp_pp_2_key(tp, pp):
    return "tp" + str(tp) + "_pp" + str(pp)


def prepare():
    id2info = {}
    for tp_index in range(TP_SIZE):
        for pp_index in range(PP_SIZE):
            key = tp_pp_2_key(tp_index, pp_index)
            if not PP_SIZE or PP_SIZE == 1:
                save_path_subfix = f"{tp_index:02d}"
            else:
                save_path_subfix = f"{tp_index:02d}_{pp_index:03d}"
            save_path = MEGATRON_SAVE_PATH + "mp_rank_" + save_path_subfix + "/model_optim_rng.pt"
            id2info[key] = {
                "save_path": save_path,
                "paras": {"model": {"language_model": {"encoder": {}}}, "checkpoint_version": 3.0},
            }
    return id2info


def load_hug_state():
    state_dict = {}
    sub_paths = [p for p in os.listdir(HUG_LOAD_PATH) if ("pytorch" in p and "bin" in p and "000" in p)]
    for sub_path in sub_paths:
        p = HUG_LOAD_PATH + sub_path
        chunk = torch.load(p, map_location="cpu")
        state_dict.update(chunk)
    return state_dict


def transform_hug_to_megatron(hug_state_dict, megatron_save_info):
    # process last_layenorm, lm_head, embedding
    for hug_key, hug_para in hug_state_dict.items():
        if "inv_freq" in hug_key:
            continue
        if "embed_tokens" in hug_key:
            hug_para = hug_para[:-256]
            pp_index = 0
            emb_chunks = torch.chunk(hug_para, TP_SIZE, 0)
            for tp_index in range(TP_SIZE):
                key = tp_pp_2_key(tp_index, pp_index)
                model_para_dict = megatron_save_info[key]["paras"]["model"]["language_model"]
                model_para_dict["embedding"] = {}
                model_para_dict["embedding"]["word_embeddings"] = {}
                model_para_dict["embedding"]["word_embeddings"]["weight"] = emb_chunks[tp_index]

        elif "lm_head.weight" == hug_key:
            pp_index = PP_SIZE - 1
            hug_para = hug_para[:-256]
            lm_head_chunks = torch.chunk(hug_para, TP_SIZE, 0)
            for tp_index in range(TP_SIZE):
                key = tp_pp_2_key(tp_index, pp_index)
                model_para_dict = megatron_save_info[key]["paras"]["model"]["language_model"]
                model_para_dict["output_layer"] = {}
                model_para_dict["output_layer"]["weight"] = lm_head_chunks[tp_index]

        elif "model.norm.weight" == hug_key:
            pp_index = PP_SIZE - 1
            for tp_index in range(TP_SIZE):
                key = tp_pp_2_key(tp_index, pp_index)
                model_para_dict = megatron_save_info[key]["paras"]["model"]["language_model"]["encoder"]
                model_para_dict["final_layernorm.weight"] = hug_para
    print("FINISH PROCESS LM_HEAD/FINAL_LAYERNORM/EMBEDDING")

    # process attention layers
    for layer_index in range(LAYER_NUM):
        print("PROCESS ATTENTION LAYER:\t" + str(layer_index))
        layer_prefix = "model.layers." + str(layer_index)
        pp_index = layer_index // (LAYER_NUM // PP_SIZE)
        q_proj = hug_state_dict[layer_prefix + ".self_attn.q_proj.weight"]
        k_proj = hug_state_dict[layer_prefix + ".self_attn.k_proj.weight"]
        v_proj = hug_state_dict[layer_prefix + ".self_attn.v_proj.weight"]
        o_proj = hug_state_dict[layer_prefix + ".self_attn.o_proj.weight"]
        up_proj = hug_state_dict[layer_prefix + ".mlp.up_proj.weight"]
        down_proj = hug_state_dict[layer_prefix + ".mlp.down_proj.weight"]
        gate_proj = hug_state_dict[layer_prefix + ".mlp.gate_proj.weight"]
        input_layernorm = hug_state_dict[layer_prefix + ".input_layernorm.weight"]
        post_attention_layernorm = hug_state_dict[layer_prefix + ".post_attention_layernorm.weight"]
        # qkv_chunks = torch.chunk(torch.concat([q_proj, k_proj, v_proj],0), TP_SIZE, 0)
        # q_chunks = torch.chunk(q_proj, TP_SIZE, 0)
        # k_chunks = torch.chunk(k_proj, TP_SIZE, 0)
        # v_chunks = torch.chunk(v_proj, TP_SIZE, 0)
        q_resize = q_proj.view(HEAD_NUM, -1, HIDDEN_DIM)
        k_resize = k_proj.view(HEAD_NUM, -1, HIDDEN_DIM)
        v_resize = v_proj.view(HEAD_NUM, -1, HIDDEN_DIM)
        qkv_resize = torch.concat([q_resize, k_resize, v_resize], 1).view(-1, HIDDEN_DIM)
        qkv_chunks = torch.chunk(qkv_resize, TP_SIZE, 0)
        o_chunks = torch.chunk(o_proj, TP_SIZE, 1)

        up_proj_resize = up_proj.view(TP_SIZE, -1, HIDDEN_DIM)
        gate_proj_resize = gate_proj.view(TP_SIZE, -1, HIDDEN_DIM)
        gate_up_resize = torch.concat([gate_proj_resize, up_proj_resize], 1).view(-1, HIDDEN_DIM)
        gate_up_chunks = torch.chunk(gate_up_resize, TP_SIZE, 0)

        down_chunks = torch.chunk(down_proj, TP_SIZE, 1)
        for tp_index in range(TP_SIZE):
            key = tp_pp_2_key(tp_index, pp_index)
            model_para_dict = megatron_save_info[key]["paras"]["model"]["language_model"]["encoder"]
            megatron_layer_index = layer_index % (LAYER_NUM // PP_SIZE)
            megatron_layer_prefix = "layers." + str(megatron_layer_index)

            # megatron_input_layernorm
            m_input_layernorm = input_layernorm
            m_input_layernorm_key = megatron_layer_prefix + ".input_layernorm.weight"
            model_para_dict[m_input_layernorm_key] = m_input_layernorm

            # megatron_post_attention_layernorm
            m_post_attention_layernorm = post_attention_layernorm
            m_post_attention_layernorm_key = megatron_layer_prefix + ".post_attention_layernorm.weight"
            model_para_dict[m_post_attention_layernorm_key] = m_post_attention_layernorm

            # megatron_qkv
            qkv_chunk = qkv_chunks[tp_index]
            # torch.concat([q_chunks[tp_index], k_chunks[tp_index], v_chunks[tp_index]], 0)
            qkv_chunk_key = megatron_layer_prefix + ".self_attention.query_key_value.weight"
            model_para_dict[qkv_chunk_key] = qkv_chunk

            # megatron output projection
            o_chunk = o_chunks[tp_index]
            o_chunk_key = megatron_layer_prefix + ".self_attention.dense.weight"
            model_para_dict[o_chunk_key] = o_chunk

            # megatron down projection
            down_proj_chunk = down_chunks[tp_index]
            down_chunk_key = megatron_layer_prefix + ".mlp.dense_4h_to_h.weight"
            model_para_dict[down_chunk_key] = down_proj_chunk

            # megatron gate_up projection
            gate_up_proj_chunk = gate_up_chunks[tp_index]
            gate_up_chunk_key = megatron_layer_prefix + ".mlp.dense_h_to_4h.weight"
            model_para_dict[gate_up_chunk_key] = gate_up_proj_chunk


def save_megatron(megatron_save_info):
    for k, v in megatron_save_info.items():
        save_path = v["save_path"]

        paras = v["paras"]
        print("SAVE: " + k + "\tTO " + save_path)
        torch.save(paras, save_path)


megatron_save_info = prepare()
hug_state_dict = load_hug_state()
print("FINISH LOAD HUGGING")
transform_hug_to_megatron(hug_state_dict, megatron_save_info)
save_megatron(megatron_save_info)
