import torch
import argparse
from utils.edit_pipeline import EditingPipeline
from utils.attention_pipeline import open_attention, process_map
import os

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=0)
    parser.add_argument('--task_name', type=str, default='cat2dog')
    parser.add_argument('--num_ddim_steps', type=int, default=15)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--save_path', type=str, default='attention_map')
    parser.add_argument('--show_F', type=bool, default=False)
    parser.add_argument('--show_SAM', type=bool, default=False)
    parser.add_argument('--show_CAM', type=bool, default=True)
    parser.add_argument('--res', type=int, default=64)
    parser.add_argument('--use_float_16', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = init_parser()
    map_dict, available_features, prompt = open_attention(args)
    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    os.makedirs(os.path.join(args.save_path, "visualization"), exist_ok=True)

    pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    tokenizer = pipe.tokenizer
    process_map(tokenizer, map_dict, available_features, args.res, prompt, args.save_path)