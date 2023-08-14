# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
import copy
import json
import os
import random
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from transformers import PreTrainedTokenizer, HfArgumentParser, PretrainedConfig
from data_processer import DataStrategy, TokenSupervision, TokenUnSupervision,TokenSupervisionRounds
from config import *
from aigc_zoo.model_zoo.rwkv4.llm_model import RwkvConfig,set_model_profile,LoraArguments,LoraConfig,PromptArguments
from aigc_zoo.model_zoo.rwkv4.rwkv4_tokenizer import RWKVTokenizer

data_conf = {
    'strategy': DataStrategy.sup,  # 数据策略选项
    DataStrategy.sup: {
        'stride':  int(train_info_args['max_seq_length'] / 3 * 2),
    },
    DataStrategy.unsup: {
        'stride':  int(train_info_args['max_seq_length'] / 3 * 2),
    },
    DataStrategy.sub_rounds: {
        'stride': int(train_info_args['max_seq_length'] / 3 * 2),
    }
}


def preprocess(text):
  return text

def postprocess(text):
  return text


class NN_DataHelper(DataHelper):
    index = 1

    def __init__(self, *args, **kwargs):
        super(NN_DataHelper, self).__init__(*args, **kwargs)
        assert data_conf[DataStrategy.sup]['stride'] > 0
        assert data_conf[DataStrategy.unsup]['stride'] > 0

    def load_tokenizer_and_config(self,*args,**kwargs):
        if 'config_class_name' not in kwargs:
            kwargs['config_class_name'] = RwkvConfig
        if 'tokenizer_class_name' not in kwargs:
            if os.path.basename(self.model_args.model_name_or_path).lower().find('world') != -1:
                kwargs['tokenizer_class_name'] = RWKVTokenizer
                tokenizer_kwargs = kwargs.get('tokenizer_kwargs',{})
                kwargs['tokenizer_kwargs'] = tokenizer_kwargs
                tokenizer_kwargs['bos_token_id'] = 0
                tokenizer_kwargs['eos_token_id'] = 0
                tokenizer_kwargs['pad_token_id'] = 1
                tokenizer_kwargs['sep_token_id'] = None

        ret = super().load_tokenizer_and_config(*args,**kwargs)
        self._preprocess_tokenizer_config()
        return ret

    def _preprocess_tokenizer_config(self):
        # model_args = self.model_args
        tokenizer = self.tokenizer
        config = self.config
        if config.decoder_start_token_id is None:
            config.decoder_start_token_id = config.bos_token_id
        assert config.decoder_start_token_id == config.bos_token_id


    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1

        tokenizer: PreTrainedTokenizer
        config = self.config
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer

        examples = data


        strategy = data_conf['strategy']
        if strategy == DataStrategy.sup:
            ds = TokenSupervision.process(tokenizer, config=config,  max_seq_length=max_seq_length, examples=examples,**data_conf[strategy])
        elif strategy == DataStrategy.unsup:
            ds = TokenUnSupervision.process(tokenizer, config=config,  max_seq_length=max_seq_length, examples=examples, **data_conf[strategy])
        elif strategy == DataStrategy.sub_rounds:
            ds = TokenSupervisionRounds.process(tokenizer, config=config, max_seq_length=max_seq_length, examples=examples,
                                            **data_conf[strategy])
        else:
            raise ValueError('Invalid strategy', strategy)
        if not ds:
            return None

        if self.index < 3:
            print(ds[0])
        return ds

    def _get_paragraph(self, lines):
        D = []
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            paragraph = jd['paragraph']
            if line_id < 10:
                print(paragraph)

            prefix = jd.get('p', '')
            paragraph = [(preprocess(session['q']),
                          preprocess('\n'.join(session['a'])) if isinstance(session['a'], list) else preprocess(
                              session['a']))
                         for session in paragraph]
            sub = []
            # 自行做模板
            for (q, a) in paragraph:
                assert len(a), ValueError('answer cannot empty')
                sub.append((q, a))
            D.append((prefix, copy.deepcopy(sub)))
            sub.clear()
        return D

    def _get_messages(self, lines):
        D = []
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            conversations = jd['conversations']
            if line_id < 10:
                print(conversations)

            paragraph = []
            prefix = ''
            pair = [None, None]
            for m in conversations:
                if m["from"] == 'user':
                    pair[0] = preprocess(m["value"])
                elif m["from"] == 'assistant':
                    pair[1] = preprocess(m["value"])
                elif m["from"] == 'system':
                    prefix = preprocess(m["value"])
                if pair[0] is not None and pair[1] is not None:
                    paragraph.append(tuple(pair))
                    pair[0], pair[1] = None, None

            sub = []
            # 自行做模板
            for (q, a) in paragraph:
                assert len(a), ValueError('answer cannot empty')
                sub.append((q, a))
            D.append((prefix, copy.deepcopy(sub)))
            sub.clear()
        return D

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            is_new = False
            if len(lines) > 0:
                is_new = 'conversations' in json.loads(lines[0])
            if is_new:
                D.extend(self._get_messages(lines))
            else:
                D.extend(self._get_paragraph(lines))
        return D

    def collate_fn(self, batch):
        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        maxlen = torch.max(o.pop('seqlen'))
        o['input_ids'] = o['input_ids'][:, :maxlen]
        o['attention_mask'] = o['attention_mask'][:, :maxlen]
        o['labels'] = o['labels'][:, :maxlen].long()
        return o

    def make_dataset_all(self):
        data_args = self.data_args

        # schema for arrow parquet
        schema = {
            "input_ids": "int32_list",
            "attention_mask": "int32_list",
            "labels": "int32_list",
            "seqlen": "int32_list",
        }
        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train',
                                        schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval', schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test', schema=schema)





if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments,PromptArguments))
    model_args, training_args, data_args, _,_ = parser.parse_dict(train_info_args)


    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16})

    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    dataHelper.make_dataset_all()

    # def shuffle_records(record_filenames, outfile, compression_type='GZIP'):
    #     print('shuffle_records record...')
    #     options = RECORD.TFRecordOptions(compression_type=compression_type)
    #     dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    #     data_size = len(dataset_reader)
    #     all_example = []
    #     for i in tqdm(range(data_size), desc='load records'):
    #         serialized = dataset_reader[i]
    #         all_example.append(serialized)
    #     dataset_reader.close()
    #
    #     shuffle_idx = list(range(data_size))
    #     random.shuffle(shuffle_idx)
    #     writer = WriterObject(outfile, options=options)
    #     for i in tqdm(shuffle_idx, desc='shuffle record'):
    #         example = all_example[i]
    #         writer.write(example)
    #     writer.close()
    #
    #
    # # 对每个record 再次打乱
    # for filename in dataHelper.train_files:
    #     shuffle_records(filename, filename)
