import os, pdb
import argparse
import numpy as np
import torch
import torch.nn as nn
import requests
from PIL import Image
import clip
import warnings

from diffusers import DDIMScheduler
from utils.edit_pipeline import EditingPipeline
from utils.target_prompt import make_target_prompt
from typing import List

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def cos_sim(x1: torch.tensor, x2: torch.tensor) -> torch.tensor:
    return torch.dot(x1, x2) / (torch.norm(x1) * torch.norm(x2))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_str', type=str, required=True)
    parser.add_argument('--random_seed', default=0)
    parser.add_argument('--task_name', type=str, default='cat2dog')
    parser.add_argument('--results_folder', type=str, default='output/test_cat')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--xa_guidance', default=0.15, type=float)
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--mask_res', type=int, default=16)
    parser.add_argument('--posterior_guidance', default=0.1, type=float)
    parser.add_argument('--mixup_type', default=['cross-attention', 'noise'], type=List[str])

    args = parser.parse_args()
   
    os.makedirs(args.results_folder, exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # make the input noise map
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    else:
        torch.manual_seed(args.random_seed)

    x = torch.randn((1, 4, 64, 64), device=device)

    # Make the editing pipeline
    pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    (src_word, tgt_word) = args.task_name.split("2")
    model, preprocess = clip.load("ViT-B/32", device=device)
    tokenized_src_word = clip.tokenize([src_word]).to(device)
    src_word_features = model.encode_text(tokenized_src_word)

    prompt_str = args.prompt_str
    tgt_prompt, idx = make_target_prompt(prompt_str, src_word, tgt_word, src_word_features, model, device)    
    print(prompt_str, '->', tgt_prompt)
    
    rec_pil, edit_pil, mask_target, mask_target_no_self = pipe(
        prompt = args.prompt_str,
        tgt_prompt = tgt_prompt,
        num_inference_steps=args.num_ddim_steps,
        x_in=x,
        guidance_scale=args.negative_guidance_scale,
        negative_prompt="", # use the empty string for the negative prompt
        task_name=args.task_name,
        mask_res=args.mask_res,
        tgt_word_start_idx=idx,
        posterior_guidance=args.posterior_guidance,
        mixup_type=args.mixup_type,
    )

    # bname = os.path.basename(inv_path).split(".")[0]
    edit_pil[0].save(os.path.join(args.results_folder, f"edit.png"))
    rec_pil[0].save(os.path.join(args.results_folder, f"reconstruction.png"))
    mask_target[0].save(os.path.join(args.results_folder, f"mask.png"))
    mask_target_no_self[0].save(os.path.join(args.results_folder, f"mask_no_attn.png"))

