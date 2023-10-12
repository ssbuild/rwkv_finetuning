# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer_lora_finetuning
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser

from data_processer import build_template
from data_utils import train_info_args, NN_DataHelper
from aigc_zoo.model_zoo.rwkv4.llm_model import MyTransformer, PetlArguments, RwkvConfig,set_model_profile
from aigc_zoo.utils.rwkv4_generate import Generate

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,)  = parser.parse_dict(train_info_args, allow_extra_keys=True)

    # 可以自行修改 RWKV_T_MAX  推理最大长度
    set_model_profile(RWKV_T_MAX=2048, RWKV_FLOAT_MODE='')

    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config()

    # 一般根据时间排序选最新的权重文件夹
    ckpt_dir = '../scripts/best_ckpt/last'

    config = RwkvConfig.from_pretrained(ckpt_dir)
    lora_args = PetlArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args, lora_args=lora_args,
                             torch_dtype=torch.float16,new_num_tokens=new_num_tokens,
                             
                             # # device_map="auto",
                             # device_map = {"":0} # 第一块卡
                             )

    # 加载lora权重
    pl_model.load_sft_weight(ckpt_dir)

    pl_model.eval().half().cuda()

    enable_merge_weight = False

    if enable_merge_weight:
        # 合并lora 权重 保存
        pl_model.save_llm_sft_weight(os.path.join(ckpt_dir, 'pytorch_model_merge.bin'), merge_lora_weight=True)
    else:
        model = pl_model.get_llm_model()

        text_list = ["写一个诗歌，关于冬天",
                     "晚上睡不着应该怎么办",
                     "从南京到上海的路线",
                     ]
        for input in text_list:
            query = build_template(input)
            response = Generate.generate(model, query=query, tokenizer=tokenizer, max_length=512,
                                         eos_token_id=config.eos_token_id,
                                         pad_token_id=config.eos_token_id,
                                         do_sample=True, top_p=0.85, temperature=1.0, )
            print('input', input)
            print('output', response)