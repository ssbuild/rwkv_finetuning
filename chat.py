# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/6/8 13:03

print('Loading...')

import numpy as np
import os, copy, types, gc, sys

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cuda.matmul.allow_tf32 = True
# np.set_printoptions(precision=4, suppress=True, linewidth=200)

# Load Model

import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser
from data_utils import train_info_args, NN_DataHelper
from models import MyTransformer, RwkvConfig,set_model_profile

parser = HfArgumentParser((ModelArguments, DataArguments))
model_args, data_args = parser.parse_dict(train_info_args, allow_extra_keys=True)

dataHelper = NN_DataHelper(model_args, None, data_args)
tokenizer, config, _,_= dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16},config_class_name=RwkvConfig)
dataHelper.preprocess_tokenizer_config()

set_model_profile(RWKV_T_MAX=config.ctx_len, RWKV_FLOAT_MODE='16')

pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=torch.float16)
model = pl_model.get_llm_model()

model.requires_grad_(False)
model.eval().half().cuda()

CHAT_LANG = 'Chinese'


if CHAT_LANG == 'English':
    user = "User"
    bot = "Bot"
    interface = ":"

    # The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.
    # The following is a conversation between a highly knowledgeable and intelligent AI called {bot}, and a human called {user}. In the following interactions, {user} and {bot} converse in natural language, and {bot} do its best to answer {user}'s questions. {bot} is respectful, polite and inclusive. {bot} knows a lot, and always tells the truth.

    init_prompt = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} french revolution what year

{bot}{interface} The French Revolution started in 1789, and lasted 10 years until 1799.

{user}{interface} 3+5=?

{bot}{interface} The answer is 8.

{user}{interface} guess i marry who ?

{bot}{interface} Only if you tell me more about yourself - what are your interests?

{user}{interface} solve for a: 9-a=2

{bot}{interface} The answer is a = 7, because 9 - 7 = 2.

{user}{interface} wat is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''
    HELP_MSG = '''Commands:
say something --> chat with bot. use \\n for new line.
+alt --> alternate chat reply
+reset --> reset chat

+gen YOUR PROMPT --> free generation with any prompt. use \\n for new line.
+qa YOUR QUESTION --> free generation - ask any question (just ask the question). use \\n for new line.
+more --> continue last free generation (only for +gen / +qa)
+retry --> retry last free generation (only for +gen / +qa)

Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B for best results.
This is not instruct-tuned for conversation yet, so don't expect good quality. Better use +gen for free generation.
'''
elif CHAT_LANG == 'Chinese':
    user = "Q"
    bot = "A"
    interface = ":"

    init_prompt = '''
Q: 企鹅会飞吗？

A: 企鹅是不会飞的。它们的翅膀主要用于游泳和平衡，而不是飞行。

Q: 西瓜是什么

A: 西瓜是一种常见的水果，是一种多年生蔓生藤本植物。西瓜的果实呈圆形或卵形，通常是绿色的，里面有红色或黄色的肉和很多的籽。西瓜味甜，多吃可以增加水分，是夏季非常受欢迎的水果之一。

'''
    HELP_MSG = '''指令:
直接输入内容 --> 和机器人聊天，用\\n代表换行
+alt --> 让机器人换个回答
+reset --> 重置对话

+gen 某某内容 --> 续写任何中英文内容，用\\n代表换行
+qa 某某问题 --> 问独立的问题（忽略上下文），用\\n代表换行
+more --> 继续 +gen / +qa 的回答
+retry --> 换个 +gen / +qa 的回答

现在可以输入内容和机器人聊天（注意它不怎么懂中文，它可能更懂英文）。请经常使用 +reset 重置机器人记忆。
'''






model_tokens = []

current_state = None


from torch.nn import functional as F
def sample_logits(out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None):
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    lastChar = int(x[-1])

    probs = F.softmax(out, dim=-1)


    top_p = top_p_usual

    if os.environ.get("RWKV_RUN_DEVICE","cuda") == "cpu":
        probs = probs.numpy()
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
        probs = probs / np.sum(probs)
        out = np.random.choice(a=len(probs), p=probs)
        return out
    else:
        sorted_probs = torch.sort(probs, descending=True)[0]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return out




########################################################################################################

def run_rnn(tokens, newline_adj=0):

    global model,model_tokens, current_state
    for i in range(len(tokens)):
        model_tokens += [int(tokens[i])]
        input_ids = torch.tensor([model_tokens[-1:]],dtype=torch.int64,device=model.device)
        if current_state is not None:
            current_state = [_.to(model.device) for _ in current_state]

        if i == len(tokens) - 1:
            o = model.forward(input_ids, state=current_state, return_dict=True)
            out = o.logits.cpu()[0][0].float()
        else:
            o = model.forward(input_ids, state=current_state, return_dict=True,return_state_only=True)
        current_state = [_.detach().cpu() for _ in o.state]

    # print(f'### model ###\n[{tokenizer.decode(model_tokens)}]')

    out[0] = -999999999  # disable <|endoftext|>
    out[187] += newline_adj
    # if newline_adj > 0:
    #     out[15] += newline_adj / 2 # '.'
    return out


all_state = {}


def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(current_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)


def load_all_stat(srv, name):
    global model_tokens, current_state
    n = f'{name}_{srv}'
    current_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']


########################################################################################################

# Run inference
print(f'\nRun prompt...')

out = run_rnn(tokenizer.encode(init_prompt))
gc.collect()
torch.cuda.empty_cache()

save_all_stat('', 'chat_init', out)

srv_list = ['dummy_server']
for s in srv_list:
    save_all_stat(s, 'chat', out)

print(f'### prompt ###\n[{tokenizer.decode(model_tokens)}]\n')


def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')


def on_message(message):
    global model_tokens, current_state

    srv = 'dummy_server'

    msg = message.replace('\\n', '\n').strip()
    if len(msg) > 1000:
        reply_msg('your message is too long (max 1000 tokens)')
        return

    x_temp = 1.0
    x_top_p = 0.85
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp=" + f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p=" + f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0

    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return

    elif msg[:5].lower() == '+gen ' or msg[:4].lower() == '+qa ' or msg.lower() == '+more' or msg.lower() == '+retry':

        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            current_state = None
            out = run_rnn(tokenizer.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qa ':
            out = load_all_stat('', 'chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')

            out = run_rnn(tokenizer.encode(new))
            save_all_stat(srv, 'gen_0', out)

            # new = f"\nThe following is an excellent Q&A session consists of detailed and factual information.\n\nQ: What is 3+5?\nA: The answer is 8.\n\nQ: {msg[9:].strip()}\nA:"
            # print(f'### prompt ###\n[{new}]')
            # current_state = None
            # out = run_rnn(tokenizer.encode(new))
            # save_all_stat(srv, 'gen_0', out)

        elif msg.lower() == '+more':
            try:
                out = load_all_stat(srv, 'gen_1')
                save_all_stat(srv, 'gen_0', out)
            except:
                return

        elif msg.lower() == '+retry':
            try:
                out = load_all_stat(srv, 'gen_0')
            except:
                return

        begin = len(model_tokens)
        out_last = begin
        for i in range(150):
            token = sample_logits(
                out,
                model_tokens,
                config.ctx_len,
                temperature=x_temp,
                top_p_usual=x_top_p,
                top_p_newline=x_top_p,
            )
            if msg[:4].lower() == '+qa ':
                out = run_rnn([token], newline_adj=-1)
            else:
                out = run_rnn([token])

            xxx = tokenizer.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
        print('\n')
        # send_msg = tokenizer.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'gen_1', out)

    else:
        if msg.lower() == '+alt':
            try:
                out = load_all_stat(srv, 'chat_pre')
            except:
                return
        else:
            out = load_all_stat(srv, 'chat')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            out = run_rnn(tokenizer.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        print(f'{bot}{interface}', end='', flush=True)
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= 30:
                newline_adj = (i - 30) / 10
            elif i <= 130:
                newline_adj = 0
            else:
                newline_adj = (i - 130) * 0.25  # MUST END THE GENERATION
            token = sample_logits(
                out,
                model_tokens,
                config.ctx_len,
                temperature=x_temp,
                top_p_usual=x_top_p,
                top_p_newline=x_top_p,
            )
            out = run_rnn([token], newline_adj=newline_adj)

            xxx = tokenizer.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:
                print(xxx, end='', flush=True)
                out_last = begin + i + 1

            send_msg = tokenizer.decode(model_tokens[begin:])
            if '\n\n' in send_msg:
                send_msg = send_msg.strip()
                break

            # send_msg = tokenizer.decode(model_tokens[begin:]).strip()
            # if send_msg.endswith(f'{user}{interface}'): # warning: needs to fix state too !!!
            #     send_msg = send_msg[:-len(f'{user}{interface}')].strip()
            #     break
            # if send_msg.endswith(f'{bot}{interface}'):
            #     send_msg = send_msg[:-len(f'{bot}{interface}')].strip()
            #     break

        # print(f'{model_tokens}')
        # print(f'[{tokenizer.decode(model_tokens)}]')

        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'chat', out)

if __name__ == '__main__':

    print(HELP_MSG)

    while True:
        msg = input(f'{user}{interface} ')
        if len(msg.strip()) > 0:
            on_message(msg)
        else:
            print('Erorr: please say something')