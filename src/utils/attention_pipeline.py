import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from einops import rearrange
from typing import List, Tuple
from PIL import Image

def get_map_type(args: argparse):
    map_type = []
    if args.show_F:
        map_type.append('F')
    if args.show_SAM:
        map_type.append('SAM')
    if args.show_CAM:
        map_type.append('CAM')
    return map_type

def open_attention(args: argparse):
    map_type = get_map_type(args)
    var_type = ['src', 'tgt']
    available_features = []
    ret = {}
    prompt = {}
    
    for map_item in map_type:
        for var_item in var_type:
            dict_name = f"{map_item}_{var_item}"
            dict_path = os.path.join(args.save_path, f"{dict_name}.pt")
            if os.path.isfile(dict_path):
                ret[dict_name] = torch.load(dict_path, map_location='cpu')
                available_features.append(dict_name)
            else:
                print(f"File {dict_path} doesn't exist.")
    
    for var_item in var_type:
        try:
            prompt[var_item] = open(os.path.join(args.save_path, f"prompt_{var_item}.txt")).read().strip()
        except:
            print(f"File prompt_{var_item}.txt doesn't exist.")

    return ret, available_features, prompt


def process_map(tokenizer, map_dict: dict, available_featuers: list, res: int, prompt: dict, save_dir: str):
    for map_type in available_featuers:
        print(map_type, map_dict[map_type].keys())
    
    up_blocks = []
    down_blocks = []

    for map_type in available_featuers:
        prompt_select = prompt[map_type[-3:]]
        tokens = tokenizer.encode(prompt_select)
        decoder = tokenizer.decode

        all_time_out  = []
        for i, t in enumerate(map_dict[map_type].keys()): # for each sample step
            for layer in map_dict[map_type][t].keys(): # for each layer
                out = []
                attention_layer = map_dict[map_type][t][layer] # size : [chans, height*width, 77]
                num_tokens = attention_layer.shape[-1]
                size = attention_layer.shape[1]
                prev_res = int(np.sqrt(size))
        
                attention_layer = attention_layer.reshape(-1, prev_res, prev_res, num_tokens)
                if prev_res != res:
                    attention_layer = rearrange(attention_layer, 'c h w t -> c t h w')
                    if prev_res < res:
                        upsample_scale = res // prev_res
                        upsample_op = nn.Upsample(scale_factor=upsample_scale, mode='bilinear')
                        attention_layer = upsample_op(attention_layer)
                    elif prev_res > res:
                        downsample_scale = prev_res // res
                        attention_layer = F.interpolate(attention_layer, scale_factor=downsample_scale, mode='bilinear')
                    attention_layer = rearrange(attention_layer, 'c t h w -> c h w t')
                    assert attention_layer.shape[1] == res
                out.append(attention_layer)

                '''
                # Extracting for each layer (only for final denoising step)
                
                if (i == 49):
                    call_visualize_atention(attention_layer, res, tokens, save_dir, decoder, i, layer)

                    if 'down_blocks' in layer:
                        down_blocks.append(attention_layer)
                        
                    if 'up_blocks' in layer:
                        up_blocks.append(attention_layer)
                '''

            out = torch.cat(out, dim = 0)
            call_visualize_atention(out, res, tokens, save_dir, decoder, i, map_type, None)

            '''
            if len(down_blocks) != 0:
                down_blocks = torch.cat(down_blocks, dim = 0)
                call_visualize_atention(down_blocks, res, tokens, save_dir, decoder, i, map_type, 'down_blocks')

            if len(up_blocks) != 0:
                up_blocks = torch.cat(up_blocks, dim = 0)
                call_visualize_atention(up_blocks, res, tokens, save_dir, decoder, i, map_type, 'up_blocks')
    
            if len(down_blocks) != 0 and len(up_blocks) != 0:
                updown_blocks = torch.cat((down_blocks, up_blocks), dim = 0)
                call_visualize_atention(updown_blocks, res, tokens, save_dir, decoder, i, map_type, 'updown_blocks')
            '''
            all_time_out.append((out.sum(0) / out.shape[0]).unsqueeze(0))
        all_time_out = torch.cat(all_time_out, dim = 0)
        call_visualize_atention(all_time_out, res, tokens, save_dir, decoder, "all", map_type, None)

def call_visualize_atention(attention_maps, res, tokens, save_dir, decoder, i, map_type, layer = None):
    new_attn = attention_maps.sum(0) / attention_maps.shape[0] # size : res, res, 77
    final_upsample = nn.Upsample(scale_factor=(256 // res), mode='bilinear')
    new_attn = new_attn.unsqueeze(0)
    new_attn = rearrange(new_attn, 'n h w t-> n t h w')
    new_attn = final_upsample(new_attn)
    new_attn = rearrange(new_attn, 'n t h w-> n h w t')
    new_attn = new_attn.squeeze(0) # size : 256, 256, 77
    new_attn = new_attn.cpu() 
    visualize_attention(new_attn, tokens, save_dir, decoder, i, map_type, layer)


def visualize_attention(attention_maps: torch.Tensor, tokens: List, save_dir: str, decoder, step: int, map_type: str, layer = None):
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i] # take one token -> size : (width, height)
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3) # size : (width, height, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = make_image_with_text(image, decoder(int(tokens[i])))
        images.append(image)
    save_images(np.stack(images, axis=0), save_dir=save_dir, step=step, map_type=map_type, layer=layer)

def make_image_with_text(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

def save_images(images, save_dir, step, num_rows=1, offset_ratio=0.02, map_type="", layer=None):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    file_name = f"{map_type}_step_{step}_{layer}.png" if layer is not None else f"{map_type}_step_{step}.png"
    pil_img = pil_img.save(os.path.join(save_dir, 'visualization', file_name))
    print(f"Saved visualized attention map to {os.path.join(save_dir, 'visualization', file_name)}")