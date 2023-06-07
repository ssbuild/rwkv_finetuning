# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/6/6 14:44
from deep_training.nlp.models.rwkv4.convert_rwkv_checkpoint_to_hf import convert_rmkv_checkpoint_to_hf_format

if __name__ == '__main__':

    # repo_id = None
    # checkpoint_file = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023.pth'
    # output_dir = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-169m'
    # size = '169M'
    # ctx_len = 1024

    # repo_id = None
    # checkpoint_file = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066.pth'
    # output_dir = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-430m'
    # size = '430M'
    # ctx_len = 1024

    # repo_id = None
    # checkpoint_file = "/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-3b-v12/RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth"
    # output_dir = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-3b-v12'
    # size = '3B'
    # ctx_len = 4096

    repo_id = None
    checkpoint_file = "/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-1b5-v12/RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth"
    output_dir = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-1b5-v12/'
    size = '1b5'
    ctx_len = 4096


    size = size.upper()
    convert_rmkv_checkpoint_to_hf_format(repo_id,checkpoint_file=checkpoint_file,
                                         output_dir=output_dir,
                                         size=size,
                                         ctx_len=ctx_len)
