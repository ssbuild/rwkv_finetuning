# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps

train_info_models = {
    'rwkv-4-pile-169m': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-169m',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-169m/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-169m',
    },
    'rwkv-4-pile-430m': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-430m',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-430m/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-430m',
    },

    'rwkv-4-pileplus-430m': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pileplus-430m',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pileplus-430m/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pileplus-430m',
    },


    'rwkv-4-pile-3b': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-3b',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-3b/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-3b',
    },
    'rwkv-4-pile-7b': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-7b',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-7b/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-7b',
    },
    'rwkv-4-pile-14b': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-14b',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-14b/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-14b',
    },

    # 中英日语
    "rwkv-4-raven-3b-v12-Eng49%-Chn49%-Jpn1%-Other1%": {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-3b-v12',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-3b-v12/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-3b-v12',
    },
    "RWKV-4-Raven-1B5-v12-Eng98%-Other2%": {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-1b5-v12',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-1b5-v12/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-1b5-v12',
    }
}


# 'target_modules': ['query_key_value'],  # bloom,gpt_neox
# 'target_modules': ["q_proj", "v_proj"], #llama,opt,gptj,gpt_neo
# 'target_modules': ['c_attn'], #gpt2
# 'target_modules': ['project_q','project_v'] # cpmant

train_target_modules_maps = {
    'moss': ['qkv_proj'],
    'chatglm': ['query_key_value'],
    'bloom' : ['query_key_value'],
    'gpt_neox' : ['query_key_value'],
    'llama' : ["q_proj", "v_proj"],
    'opt' : ["q_proj", "v_proj"],
    'gptj' : ["q_proj", "v_proj"],
    'gpt_neo' : ["q_proj", "v_proj"],
    'gpt2' : ['c_attn'],
    'cpmant' : ['project_q','project_v'],
    'rwkv' : ['key','value','receptance'],
}
