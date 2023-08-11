import os

import torch
from tqdm import tqdm

HUG_LOAD_TEMPLATE_PATH = "HUG_LOAD_TEMPLATE_PATH"
HUG_SAVE_PATH = "HUG_SAVE_PATH"
MEGATRON_LOAD_PATH = "MEGATRON_LOAD_PATH"

PP_SIZE = 1
TP_SIZE = 2
LAYER_NUM = 36
HEAD_NUM = 36
HIDDEN_DIM = 4536
NUM_QUERY_GROUP = 2


def prepare_v2():
    id2info = {}
    id2info["word_embedding"] = {str(idx): {} for idx in range(TP_SIZE)}
    id2info["transformers"] = {
        "layer" + str(idx): {str(tp_index): {} for tp_index in range(TP_SIZE)} for idx in range(LAYER_NUM)
    }
    id2info["final_layernorm"] = {}
    id2info["lm_head"] = {str(idx): {} for idx in range(TP_SIZE)}
    for tp_index in range(TP_SIZE):
        for pp_index in range(PP_SIZE):
            if not PP_SIZE or PP_SIZE == 1:
                save_path_subfix = f"{tp_index:02d}"
            else:
                save_path_subfix = f"{tp_index:02d}_{pp_index:03d}"
            save_path = MEGATRON_LOAD_PATH + "mp_rank_" + save_path_subfix + "/model_optim_rng.pt"
            meg_state_dict = torch.load(save_path, map_location="cpu")
            encoder = meg_state_dict["model"]["language_model"]["encoder"]
            for k, v in encoder.items():
                if "final_layernorm" in k:
                    continue
                sublayer_index = int(k.split(".")[1])
                layer_index = pp_index * LAYER_NUM // PP_SIZE + sublayer_index
                para_dict = id2info["transformers"]["layer" + str(layer_index)][str(tp_index)]
                para_dict[k] = v
            if pp_index == 0:
                word_embedding = meg_state_dict["model"]["language_model"]["embedding"]["word_embeddings"]["weight"]
                id2info["word_embedding"][str(tp_index)] = word_embedding
            if pp_index == PP_SIZE - 1:
                lm_head = meg_state_dict["model"]["language_model"]["output_layer"]["weight"]
                id2info["lm_head"][str(tp_index)] = lm_head
                final_layernorm = meg_state_dict["model"]["language_model"]["encoder"]["final_layernorm.weight"]
                id2info["final_layernorm"] = final_layernorm
    return id2info


def load_hug_state():
    state_dict = {}
    sub_paths = [p for p in os.listdir(HUG_LOAD_TEMPLATE_PATH) if ("pytorch" in p and "bin" in p and "000" in p)]

    for sub_path in tqdm(sub_paths):
        p = HUG_LOAD_TEMPLATE_PATH + sub_path
        chunk = torch.load(p, map_location="cpu")
        state_dict.update(chunk)
    return state_dict


def transform_meg_to_hug(meg_state_info, hug_state_dict):
    # process last_layenorm, lm_head, embedding
    saved_hug_state_dict = {}
    for hug_key, hug_para in hug_state_dict.items():
        if "inv_freq" in hug_key:
            continue

        if "embed_tokens" in hug_key:
            # hug_emb_
            meg_paras = meg_state_info["word_embedding"]
            print("RAW_EMBEDDING_SIZE:\t", hug_para.size())
            meg_emb_list = [meg_paras[str(tp_index)] for tp_index in range(TP_SIZE)]
            meg_emb_concat = torch.concat(meg_emb_list, 0)
            print("MEG_EMBEDDING_SIZE:\t", meg_emb_concat.size())
            saved_hug_state_dict[hug_key] = meg_emb_concat

            print(hug_key, hug_para.mean(), meg_emb_concat.mean())
            """
            hug_para = hug_para[:-256]
            pp_index = 0
            emb_chunks = torch.chunk(hug_para, TP_SIZE, 0)
            for tp_index in range(TP_SIZE):
                key = tp_pp_2_key(tp_index, pp_index)
                model_para_dict = megatron_save_info[key]['paras']['model']['language_model']
                model_para_dict['embedding'] = {}
                model_para_dict['embedding']['word_embeddings'] = {}
                model_para_dict['embedding']['word_embeddings']['weight'] = emb_chunks[tp_index]
            """
        elif "lm_head.weight" == hug_key:
            meg_paras = meg_state_info["lm_head"]
            print("RAW_LM_HEAD_SIZE:\t", hug_para.size())
            meg_emb_list = [meg_paras[str(tp_index)] for tp_index in range(TP_SIZE)]
            meg_emb_concat = torch.concat(meg_emb_list, 0)
            print("MEG_LM_HEAD_SIZE:\t", meg_emb_concat.size())
            saved_hug_state_dict[hug_key] = meg_emb_concat

            print(hug_key, hug_para.mean(), meg_emb_concat.mean())
            """
            pp_index = PP_SIZE - 1
            hug_para = hug_para[:-256]
            lm_head_chunks = torch.chunk(hug_para, TP_SIZE, 0)
            for tp_index in range(TP_SIZE):
                key = tp_pp_2_key(tp_index, pp_index)
                model_para_dict = megatron_save_info[key]['paras']['model']['language_model']
                model_para_dict['output_layer'] = {}
                model_para_dict['output_layer']['weight'] = lm_head_chunks[tp_index]
            """

        elif "model.norm.weight" == hug_key:
            meg_final_layernorm = meg_state_info["final_layernorm"]
            print("RAW_layernorm_SIZE:\t", hug_para.size())
            print("MEG_layernorm_SIZE:\t", meg_final_layernorm.size())
            saved_hug_state_dict[hug_key] = meg_final_layernorm
            print(hug_key, hug_para.mean(), meg_final_layernorm.mean())
            """
            pp_index = PP_SIZE - 1
            for tp_index in range(TP_SIZE):
                key = tp_pp_2_key(tp_index, pp_index)
                model_para_dict = megatron_save_info[key]['paras']['model']['language_model']['encoder']
                model_para_dict['final_layernorm.weight'] = hug_para
            """

    print("FINISH PROCESS LM_HEAD/FINAL_LAYERNORM/EMBEDDING")

    # process attention layers
    for layer_index in range(LAYER_NUM):
        print("PROCESS ATTENTION LAYER:\t" + str(layer_index))
        hug_layer_name_prefix = "model.layers." + str(layer_index)
        meg_paras = meg_state_info["transformers"]["layer" + str(layer_index)]
        meg_layer_index = layer_index % (LAYER_NUM // PP_SIZE)
        # pp_index = layer_index // (LAYER_NUM // PP_SIZE)
        """
        q_proj = hug_state_dict[layer_prefix + '.self_attn.q_proj.weight']
        k_proj = hug_state_dict[layer_prefix + '.self_attn.k_proj.weight']
        v_proj = hug_state_dict[layer_prefix + '.self_attn.v_proj.weight']
        o_proj = hug_state_dict[layer_prefix + '.self_attn.o_proj.weight']
        up_proj = hug_state_dict[layer_prefix + '.mlp.up_proj.weight']
        down_proj = hug_state_dict[layer_prefix + '.mlp.down_proj.weight']
        gate_proj = hug_state_dict[layer_prefix + '.mlp.gate_proj.weight']
        input_layernorm = hug_state_dict[layer_prefix + '.input_layernorm.weight']
        post_attention_layernorm = hug_state_dict[layer_prefix + '.post_attention_layernorm.weight']
        """
        # recover qkv

        meg_qkvs = torch.concat(
            [
                meg_paras[str(tp_index)]["layers." + str(meg_layer_index) + ".self_attention.query_key_value.weight"]
                for tp_index in range(TP_SIZE)
            ],
            0,
        )

        HEAD_DIM = int(HIDDEN_DIM // HEAD_NUM)

        print(HEAD_DIM)

        # shape = (HEAD_NUM + 2 * NUM_QUERY_GROUP) * HEAD_DIM

        # print(shape)

        # print(meg_qkvs.size())
        # meg_qkvs_3d = meg_qkvs.view(HEAD_NUM + 2 * NUM_QUERY_GROUP, 1, HEAD_DIM, HIDDEN_DIM)
        # print(meg_qkvs_3d.size())

        # q, k, v = torch.split(meg_qkvs_3d, (HEAD_NUM, NUM_QUERY_GROUP, NUM_QUERY_GROUP), dim=0)

        q, k, v = tensor_split(meg_qkvs, NUM_QUERY_GROUP, HEAD_DIM)

        print(q.size())
        print(k.size())
        print(v.size())
        q = torch.reshape(q, [-1, HIDDEN_DIM])
        k = torch.reshape(k, [-1, HIDDEN_DIM])
        v = torch.reshape(v, [-1, HIDDEN_DIM])
        # q = q.view(-1, HIDDEN_DIM)
        # k = k.view(-1, HIDDEN_DIM)
        # v = v.view(-1, HIDDEN_DIM)
        saved_hug_state_dict[hug_layer_name_prefix + ".self_attn.q_proj.weight"] = q
        saved_hug_state_dict[hug_layer_name_prefix + ".self_attn.k_proj.weight"] = k
        saved_hug_state_dict[hug_layer_name_prefix + ".self_attn.v_proj.weight"] = v

        # recover gate_up

        meg_gate_up = torch.concat(
            [
                meg_paras[str(tp_index)]["layers." + str(meg_layer_index) + ".mlp.dense_h_to_4h.weight"]
                for tp_index in range(TP_SIZE)
            ],
            0,
        )
        print(meg_gate_up.size())
        meg_gate_up_3d = meg_gate_up.view(TP_SIZE, 2, -1, HIDDEN_DIM)
        print(meg_gate_up_3d.size())

        # meg_gate_up_3d = meg_gate_up.view(-1, 2, HIDDEN_DIM)
        gate, up = torch.chunk(meg_gate_up_3d, 2, 1)
        print(gate.size())
        gate = torch.reshape(gate, [-1, HIDDEN_DIM])
        print(gate.size())
        up = torch.reshape(up, [-1, HIDDEN_DIM])
        saved_hug_state_dict[hug_layer_name_prefix + ".mlp.gate_proj.weight"] = gate
        saved_hug_state_dict[hug_layer_name_prefix + ".mlp.up_proj.weight"] = up

        # recover layernorm
        meg_input_layernorm = meg_paras["0"]["layers." + str(meg_layer_index) + ".input_layernorm.weight"]
        meg_post_attention_layernorm = meg_paras["0"][
            "layers." + str(meg_layer_index) + ".post_attention_layernorm.weight"
        ]
        saved_hug_state_dict[hug_layer_name_prefix + ".input_layernorm.weight"] = meg_input_layernorm
        saved_hug_state_dict[hug_layer_name_prefix + ".post_attention_layernorm.weight"] = meg_post_attention_layernorm

        # recover down

        meg_down = torch.concat(
            [
                meg_paras[str(tp_index)]["layers." + str(meg_layer_index) + ".mlp.dense_4h_to_h.weight"]
                for tp_index in range(TP_SIZE)
            ],
            1,
        )
        saved_hug_state_dict[hug_layer_name_prefix + ".mlp.down_proj.weight"] = meg_down

        # recover output_proj
        meg_output = torch.concat(
            [
                meg_paras[str(tp_index)]["layers." + str(meg_layer_index) + ".self_attention.dense.weight"]
                for tp_index in range(TP_SIZE)
            ],
            1,
        )
        saved_hug_state_dict[hug_layer_name_prefix + ".self_attn.o_proj.weight"] = meg_output

        """
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
            model_para_dict = megatron_save_info[key]['paras']['model']['language_model']['encoder']
            megatron_layer_index = layer_index % (LAYER_NUM // PP_SIZE)
            megatron_layer_prefix = "layers." + str(megatron_layer_index)

            # megatron_input_layernorm
            m_input_layernorm = input_layernorm
            m_input_layernorm_key = megatron_layer_prefix + '.input_layernorm.weight'
            model_para_dict[m_input_layernorm_key] = m_input_layernorm

            # megatron_post_attention_layernorm
            m_post_attention_layernorm = post_attention_layernorm
            m_post_attention_layernorm_key = megatron_layer_prefix + '.post_attention_layernorm.weight'
            model_para_dict[m_post_attention_layernorm_key] = m_post_attention_layernorm

            # megatron_qkv
            qkv_chunk = qkv_chunks[tp_index]
            # torch.concat([q_chunks[tp_index], k_chunks[tp_index], v_chunks[tp_index]], 0)
            qkv_chunk_key = megatron_layer_prefix + '.self_attention.query_key_value.weight'
            model_para_dict[qkv_chunk_key] = qkv_chunk

            # megatron output projection
            o_chunk = o_chunks[tp_index]
            o_chunk_key = megatron_layer_prefix + '.self_attention.dense.weight'
            model_para_dict[o_chunk_key] = o_chunk

            # megatron down projection
            down_proj_chunk = down_chunks[tp_index]
            down_chunk_key = megatron_layer_prefix + '.mlp.dense_4h_to_h.weight'
            model_para_dict[down_chunk_key] = down_proj_chunk

            # megatron gate_up projection
            gate_up_proj_chunk = gate_up_chunks[tp_index]
            gate_up_chunk_key = megatron_layer_prefix + '.mlp.dense_h_to_4h.weight'
            model_para_dict[gate_up_chunk_key] = gate_up_proj_chunk
        """
    return saved_hug_state_dict


def save_hug(hug_state_dict):
    save_path = HUG_SAVE_PATH + "pytorch_model.bin"
    torch.save(hug_state_dict, save_path)


def tensor_split(param, n_query_groups, head_size):
    def kstart(start, blen, klen) -> int:
        """returns start index of keys in batch"""
        return start + (blen - (klen * 2))

    def vstart(start, blen, klen) -> int:
        """returns start index of values in batch"""
        return start + blen - klen

    def vend(start, blen) -> int:
        """returns last index of values in batch"""
        return start + blen

    # num observations
    nobs = param.shape[0]
    # batch length
    blen = nobs // n_query_groups
    # key length in batch
    klen = head_size
    # value length in batch
    vlen = head_size
    # the starting index of each new batch
    starts = range(0, nobs, blen)
    # the indices to splice on
    splices = [(s, kstart(s, blen, klen), vstart(s, blen, vlen), vend(s, blen)) for s in starts]

    qc = ()
    kc = ()
    vc = ()

    for splice in splices:
        qs, ks, vs, ve = splice
        qc += (param[qs:ks, :],)
        kc += (param[ks:vs, :],)
        vc += (param[vs:ve, :],)

    q = torch.cat(qc)
    k = torch.cat(kc)
    v = torch.cat(vc)

    return q, k, v


megatron_load_info_plus_para = prepare_v2()
# print(megatron_load_info_plus_para.keys())
hug_template_state = load_hug_state()
# print(hug_template_state.keys())
print("FINISH LOAD HUGGING")
hug_state_dict = transform_meg_to_hug(megatron_load_info_plus_para, hug_template_state)
save_hug(hug_state_dict)
