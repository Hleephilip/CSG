# Conditional Score Guidance for Text-Driven Image-to-Image Translation

Implementation of "Conditional Score Guidance for Text-Driven Image-to-Image Translation".

### Edit synthetic images

You can generate an image using Stable Diffusion and edit with CSG. Note that `posterior_guidance` is a hyperparamter related to guidance scale.

```
python src/edit_synthetic_images.py \
    --results_folder "output/synth_edit" \
    --prompt_str "a high resolution painting of a cat eating a hamburger" \
    --task "cat2squirrel" \
    --random_seed 0 \
    --mask_res 16 --posterior_guidance <GUIDANCE_SCALE>
```

Synthesized and edited images are saved in `output/synth_edit` directory.

### Edit real images

You can edit a real image with CSG. First, do the DDIM inversion using the command :

```
python src/inversion.py \
    --input_image "data/cat.png" \
    --results_folder "output/test_cat"
```

Then, perform image editing. 

```
python src/edit_real_images.py \
    --inversion "output/test_cat/inversion/cat.pt" \
    --prompt "output/test_cat/prompt/cat.txt" \
    --task_name "cat2dog" \
    --results_folder "output/test_cat/" \
    --mask_res 16 --posterior_guidance <GUIDANCE_SCALE>
```

After all, files at directory `output/test_cat` is like:

```
output/test_cat
  ├── inversion
  │   ├── cat_1.pt
  │   └── ...
  ├── prompt
  │   ├── cat_1.txt
  │   └── ...
  ├── edit
  │   ├── cat_1.png
  │   └── ...
  ├── mask_no_attn
  │   ├── cat_1.png
  │   └── ...
  ├── mask
  │   ├── cat_1.png
  │   └── ...
  └── reconstruction
      ├── cat_1.png
      └── ...
 ```
 
Reconstructed image from DDIM inversion is saved in `reconstruction/`, and edited image is in `edit/` directory. You can also check content mask in `mask_no_attn/` and smoothed content mask in `mask/`.

### Visualize cross-attention maps

You can save and visualize cross-attention maps using synthetic images. Firstly, run `save_attention_synth.py` to save the cross-attention maps:

```
python src/save_attention_synth.py \
    --results_folder "output/synth_edit" \
    --prompt_str "a high resolution painting of a cat eating a hamburger" \
    --task "cat2squirrel" \
    --random_seed 0 \
    --mask_res 16 --posterior_guidance <GUIDANCE_SCALE> \
    --save_path "attention_map"
```
Then, run `visualize_attention.py` to visualize the saved cross-attention map:

```
python src/visualize_attention.py
```

### Requirements

Refer to requirements.txt.

```
pip install -r requirements.txt
```
### Acknowledgements

This method is implemented based on [pix2pix-zero](https://github.com/pix2pixzero/pix2pix-zero/) and [prompt-to-prompt](https://github.com/google/prompt-to-prompt).
