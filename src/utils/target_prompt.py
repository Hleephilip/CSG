import clip
import torch
import numpy as np

def cos_sim(x1: torch.tensor, x2: torch.tensor) -> torch.tensor:
    return torch.dot(x1, x2) / (torch.norm(x1) * torch.norm(x2))

def make_target_prompt(prompt_str: str, src_word: str, tgt_word: str, src_word_features, model, device):
    src_prompt_split = prompt_str.split()
    try :
        idx = src_prompt_split.index(src_word)
        src_prompt_split[idx] = tgt_word
        tgt_prompt = " ".join(src_prompt_split)
    except ValueError:
        tokenized_prompt = clip.tokenize(src_prompt_split).to(device)
        prompt_features = model.encode_text(tokenized_prompt)
        sim_lst = []
        for i in range(prompt_features.shape[0]):
            sim_lst.append(cos_sim(src_word_features[0], prompt_features[i]).item())
        idx = np.argmax(sim_lst)
        del sim_lst
        
        if src_word not in src_prompt_split[idx] :
            src_prompt_split[idx] = tgt_word
        else :
            src_prompt_split[idx] = src_prompt_split[idx].replace(src_word, tgt_word)
            
        tgt_prompt = " ".join(src_prompt_split)
    return tgt_prompt, idx