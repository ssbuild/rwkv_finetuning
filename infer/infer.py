# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser

from data_processer import build_template
from data_utils import train_info_args, NN_DataHelper
from aigc_zoo.model_zoo.rwkv4.llm_model import MyTransformer, RwkvConfig,set_model_profile
from aigc_zoo.utils.rwkv4_generate import Generate


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, ))
    (model_args,)  = parser.parse_dict(train_info_args, allow_extra_keys=True)

    # 可以自行修改 RWKV_T_MAX  推理最大长度
    set_model_profile(RWKV_T_MAX=2048, RWKV_FLOAT_MODE='')

    dataHelper = NN_DataHelper(model_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16})
    
    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=torch.float16)
    model = pl_model.get_llm_model()

    model.requires_grad_(False)
    model.eval().half().cuda()



    text_list = ["你是谁?",
                 "你会干什么?",
                 ]
    for input in text_list:
        #query = build_template(input)
        response = Generate.generate(model, query=input, tokenizer=tokenizer, max_length=512,
                                          eos_token_id=config.eos_token_id,
                                          pad_token_id=config.eos_token_id,
                                          do_sample=True, top_p=0.85, temperature=1.0, )
        print('input', input)
        print('output', response)