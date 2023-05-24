# CSG

Implementation of "Conditional Score Guidance for Text-Driven Image-to-Image Translation".

### Edit synthetic images

You can generate an image using Stable Diffusion and edit with CSG.

```
python src/edit_synthetic_images.py \
    --results_folder "output/synth_edit" \
    --prompt_str "a high resolution painting of a cat eating a hamburger" \
    --task "cat2squirrel" 
    --random_seed 0 \
    --posterior_guidance 5.0
```

### Edit real images

You can edit a real image with CSG. First, do the DDIM inversion using the command :

```
python src/inversion.py \
    --input_image "data/cat.png" \
    --results_folder "output/test_cat"
```

Then, edit perform image editing.

```
python src/edit_real_ours_v8_abalation.py \
    --inversion "output/test_cat/inversion/" \
    --prompt "output/test_cat/prompt/" \
    --task_name "cat2dog" \
    --results_folder "output/test_cat/" \
    --mask_res 16 --posterior_guidance 15
```

### Requirements

Refer to [requirements.txt](https://github.com/frogyunmax/CSG/blob/main/requirements.txt)

```
pip install -r requirements.txt
```
