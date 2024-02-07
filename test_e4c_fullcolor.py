import torch
from diffusers import DiffusionPipeline 
### NOTE: consider to change into stable diffusion pipeline to avoid safety checker
from pipelines.stable_diffusion_pipeline import StableDiffusionPipeline 
from colornet_utils import create_image_with_shapes, _compute_cosine, ColorNet, ColorEmbed, ColorNet_Embed, optim_init_colornet

import os
from datetime import datetime
import argparse

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--exp", type=str, default='colorpeel_e4c_10000steps_fullcolor')
	parser.add_argument("--inf_steps", type=int, default=25)
	parser.add_argument("--seeds", type=int, default=42)
	parser.add_argument("--scale", type=float, default=6.0)
	parser.add_argument("--interval", type=float, default=0.05)
	parser.add_argument("--samples", type=int, default=3)
	parser.add_argument('--object', type=str, default='chair')

	args = parser.parse_args()
	
	prompts = [
		"a photo of {object_placeholder} in <c*> color",
		]

	model_path = f'models/{args.exp}/'
	LOG_DIR=f''

	model_id = f"{model_path+LOG_DIR}"

	out_path = f'output/{args.exp}_{args.inf_steps}_{args.seeds}_{args.scale}_{args.samples}_personlized/'

	isExist = os.path.exists(out_path)
	if not isExist:
  		os.makedirs(out_path)
		
	osExist = os.path.exists(f'{out_path}/{LOG_DIR}')

	if not osExist:
  		os.makedirs(f'{out_path}/{LOG_DIR}')
	
	# pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
	pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32).to("cuda")
	### NOTE: disable the safety checker
	pipe.safety_checker = None

	pipe.unet.load_attn_procs(f"{model_id}", weight_name="pytorch_custom_diffusion_weights.bin")

	pipe.load_textual_inversion(f"{model_id}", weight_name="<s1*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<s2*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<s3*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<s4*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<c*>.bin")
	
	### NOTE: new code to import the text encoder
	text_encoder = pipe.text_encoder
	tokenizer = pipe.tokenizer

	token_embeds = text_encoder.get_input_embeddings().weight.data

	color_embedder=ColorEmbed(num_labels=4)
	color_encoder = ColorNet_Embed()
	color_enc_path=f"{model_path}/color_encoder.pth"
	color_encoder.load_state_dict(torch.load(color_enc_path))
	color_encoder = color_encoder.cuda()
	
	color_emb_path=f"{model_path}/color_embedder.pth"
	color_embedder.load_state_dict(torch.load(color_emb_path))
	color_embedder = color_embedder.cuda()

	color_token='<c*>'
	color_token_id = tokenizer.convert_tokens_to_ids(color_token)
	### NOTE: put the new embedding over here.
	for prompt_ in prompts:
		prompt = prompt_.format(object_placeholder=args.object)
		path = f'{out_path}/{LOG_DIR}/{args.object}_{datetime.now()}'
		os.mkdir(path)
		for red in range(4):
			for green in range(4):
				for blue in range(4):
					color_fill_embed= torch.tensor([red,green,blue], dtype=torch.long).cuda()
					color_embed_pre = color_embedder(color_fill_embed)
					post_color_embed = color_encoder(color_embed_pre)

					token_embeds[color_token_id]=post_color_embed
					gen = torch.Generator(device="cuda").manual_seed(args.seeds)
					out = pipe(prompt, num_inference_steps=args.inf_steps, generator=gen, guidance_scale=args.scale,eta=1.0,)
					out.images[0].save(f"{path}/{args.object}_{red*32}_{green*32}_{blue*32}.png")

if __name__ == "__main__":
    main()