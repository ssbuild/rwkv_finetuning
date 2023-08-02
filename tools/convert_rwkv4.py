# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/6/6 14:44
import os.path

from deep_training.nlp.models.rwkv4.convert_rwkv_checkpoint_to_hf import convert_rmkv_checkpoint_to_hf_format


def convert(repo_id,checkpoint_file,output_dir,tokenizer_file,size,ctx_len,vocab_size):
    size = size.upper()
    is_world = os.path.basename(checkpoint_file).upper().find('WORLD') != -1
    convert_rmkv_checkpoint_to_hf_format(repo_id, checkpoint_file=checkpoint_file,
                                         output_dir=output_dir,
                                         tokenizer_file=tokenizer_file,
                                         size=size,
                                         ctx_len=ctx_len,
                                         vocab_size=vocab_size,
                                         is_world=is_world)
if __name__ == '__main__':

    # repo_id = None
    # checkpoint_file = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023.pth'
    # output_dir = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-169m'
    # size = '169M'
    # ctx_len = 1024
    # vocab_size = 50277
    # tokenizer_file = "20B_tokenizer.json"


    # repo_id = None
    # checkpoint_file = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066.pth'
    # output_dir = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-430m'
    # size = '430M'
    # ctx_len = 1024
    # vocab_size = 50277
    # tokenizer_file = "20B_tokenizer.json"

    # repo_id = None
    # checkpoint_file = "/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-3b-v12/RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth"
    # output_dir = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-3b-v12'
    # size = '3B'
    # ctx_len = 4096
    # vocab_size = 50277
    # tokenizer_file = "20B_tokenizer.json"

    # repo_id = None
    # checkpoint_file = "/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-1b5-v12/RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth"
    # output_dir = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-1b5-v12/'
    # size = '1b5'
    # ctx_len = 4096
    # vocab_size = 50277
    # tokenizer_file = "20B_tokenizer.json"




    repo_id = None
    checkpoint_file = "/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-World-CHNtuned-0.4B-v1/RWKV-4-World-CHNtuned-0.4B-v1-20230618-ctx4096.pth"
    output_dir = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-World-CHNtuned-0.4B-v1'
    size = '0.4b'
    ctx_len = 4096
    vocab_size = 65536
    tokenizer_file = "rwkv_vocab_v20230424.txt"

    convert(repo_id, checkpoint_file, output_dir, tokenizer_file, size, ctx_len, vocab_size)

    #不建议使用CHNtuned-1.5B
    repo_id = None
    checkpoint_file = "/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-World-CHNtuned-1.5B-v1/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096.pth"
    output_dir = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-World-CHNtuned-1.5B-v1'
    size = '1b5'
    ctx_len = 4096
    vocab_size = 65536
    tokenizer_file = "rwkv_vocab_v20230424.txt"

    convert(repo_id, checkpoint_file, output_dir, tokenizer_file, size, ctx_len, vocab_size)

    repo_id = None
    checkpoint_file = "/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-World-CHNtuned-3B-v1/RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096.pth"
    output_dir = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-World-CHNtuned-3B-v1'
    size = '3B'
    ctx_len = 4096
    vocab_size = 65536
    tokenizer_file = "rwkv_vocab_v20230424.txt"

    convert(repo_id,checkpoint_file,output_dir,tokenizer_file,size,ctx_len,vocab_size)



