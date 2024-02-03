import argparse
import hashlib
import itertools
import cv2
import json
import logging
import matplotlib.pyplot as plt
import math
import os
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfApi, create_repo
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    # UNet2DConditionModel,
)

from custom_attention.unet_2d_condition_custom import UNet2DConditionModel

from custom_attention.loaders_custom import AttnProcsLayers
# from diffusers.models.attention_processor import CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor
from custom_attention.attention_processor_custom import CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor

from colornet_utils import create_image_with_shapes, _compute_cosine

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from IPython.display import display

import wandb


logger = get_logger(__name__)

shape_id_dict = {'circle':0, 'square':1, 'triangle':2, 'hexagon':3}
shape_id_dict_reverse = {v: k for k, v in shape_id_dict.items()}

shape_token_dict = {'circle':'<s1*>', 'square':'<s2*>', 'triangle':'<s3*>', 'hexagon':'<s4*>'}
shape_token_dict_reverse = {v: k for k, v in shape_id_dict.items()}

color_id_dict = {'red':0, 'green':1, 'blue':2, 'yellow':3}
color_id_dict_reverse = {0:(255,0,0), 1:(0,255,0), 2:(0,0,255), 3:(255,255,0)}

color_templates = [
    "a photo of {shape_token} in {color_token}",
    "a photo of {shape_token} shape in {color_token} color",
    "a photo of {shape_token} filled with {color_token}",
    "a photo of {shape_token} {color_token}",
]



def freeze_params(params):
    for param in params:
        param.requires_grad = False

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, fontScale=1, thickness=2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img

def show_cross_attention_blackwhite(prompts, attention_maps, display_image=False,):
    images = []
    split_imgs = []
    for i in range(len(prompts)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        split_imgs.append(image)
        
        image = text_under_image(image, prompts[i])
        images.append(image)
    pil_img=view_images(np.stack(images, axis=0),display_image=display_image)
    return pil_img, split_imgs

def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = False) -> Image.Image:
    """ Displays a list of images in a grid. """
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
    if display_image:
        display(pil_img)
    return pil_img


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def collate_fn(examples, with_prior_preservation):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    color_fill_embed = [example["color_fill_embed"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        mask += [example["class_mask"] for example in examples]

    input_ids = torch.cat(input_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    mask = torch.stack(mask)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()
    color_fill_embed = torch.cat(color_fill_embed, dim=0)

    batch = {"input_ids": input_ids, "pixel_values": pixel_values, \
             "mask": mask.unsqueeze(1), "color_fill_embed": color_fill_embed}
    return batch



### NOTE: modify this class for color encoder
class CustomDiffusionDataset(Dataset):
    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        mask_size=64,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        aug=False,
        shape_size=256,
    ):
        self.shape_size=shape_size
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        for concept in concepts_list:
            inst_img_path = [
                (x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()
            ]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
                self.class_images_path.extend(class_img_path[:num_class_images])

        # random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        # self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                # self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top : top + inner, left : left + inner, :] = image
            mask[
                top // factor + 1 : (top + scale) // factor - 1, left // factor + 1 : (left + scale) // factor - 1
            ] = 1.0

        return instance_image, mask

    def __getitem__(self, index):
        example = {}
        ### NOTE: automatical generation
        start_color, end_color = torch.tensor([200,0,0], dtype=torch.float32), torch.tensor([255,0,0], dtype=torch.float32)
        ### NOTE: hard coding
        self.start_color_embed, self.end_color_embed = torch.zeros(1,768), torch.ones(1,768)

        color_lambda = torch.rand(1)
        color_fill = (start_color * (1-color_lambda) + end_color * color_lambda).to(torch.int)
        color_fill_embed = (self.start_color_embed * (1-color_lambda) + self.end_color_embed * color_lambda)

        shape_label = torch.randint(0, 4, (1,))
        shape = shape_id_dict_reverse[shape_label.item()]
        shape_token=shape_token_dict[shape]

        color_fill_tuple = tuple(color_fill.tolist())

        instance_image = create_image_with_shapes(circle_diameter=self.shape_size, \
                                          fill_color=color_fill_tuple, shape=shape)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)
        
        template_ = random.choices(color_templates, k=1)[0]
        instance_prompt_idx = template_.format(color_token="<c*>", shape_token=shape_token)
        
        example["color_fill_embed"] = color_fill_embed
        
        example["instance_prompt_ids"] = self.tokenizer(
            # random.choice(instance_prompt),
            instance_prompt_idx,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example

def save_new_embed(text_encoder, modifier_token_id, accelerator, args, output_dir):
    """Saves the new token embeddings from the text encoder."""

    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight

    # q_emb = []
    # lc_emb = []

    for x, y in zip(modifier_token_id, args.modifier_token):
        learned_embeds_dict = {}
        learned_embeds_dict[y] = learned_embeds[x]
        torch.save(learned_embeds_dict, f"{output_dir}/{y}.bin")


def decode_image(vae_, latents_):
    dlat = 1 / vae_.config.scaling_factor * latents_
    dimg = vae_.decode(dlat)['sample']
    dimg = (dimg / 2 + 0.5).clamp(0, 1)
    dimg = dimg.cpu().permute(0, 2, 3, 1).detach().numpy()
    dimg = (dimg * 255).astype(np.uint8)

    return dimg

def visualize_attn(embed_, step, ssaa):
	embed_ = embed_.detach().cpu()
	num_channels = embed_.shape[2]
	num_rows = 10
	num_cols = 8
	
	fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 15))
	
	for i in range(num_channels):
		row, col = i // num_cols, i % num_cols
		ax = axes[row, col]
		
		channel_image = embed_[:, :, i]
		ax.imshow(channel_image, cmap='gray')
		ax.axis('off')
		
	for i in range(num_channels, num_rows * num_cols):
		row, col = i // num_cols, i % num_cols
		axes[row, col].axis('off')
		
	plt.tight_layout()
	plt.show()
	plt.savefig(f'{args.output_dir}/attn_maps/output_{step}_{ssaa}.png', facecolor='black')
