import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from itertools import islice
from einops import rearrange
from torch import autocast
import base64
from io import BytesIO

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

torch.set_grad_enabled(False)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


model = None
sampler = None
# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global sampler

    config = OmegaConf.load(f"configs/stable-diffusion/v2-inference.yaml")
    model = load_model_from_config(config, f"stable-diffusion-2/768-v-ema.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global sampler

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 768)
    width = model_inputs.get('width', 768)
    num_inference_steps = model_inputs.get('num_inference_steps', 50)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    input_seed = model_inputs.get("seed",None)
    
    #If "seed" is not sent, we won't specify a seed in the call
    generator = None
    if input_seed != None:
        generator = torch.Generator("cuda").manual_seed(input_seed)
    
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    image = None
    with torch.no_grad(), autocast("cuda"):
        for prompts in tqdm([[prompt]], desc="data"):
            uc = None
            if guidance_scale != 1.0:
                uc = model.get_learned_conditioning([""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            c = model.get_learned_conditioning(prompts)
            shape = [4, height // 8, width// 8]
            samples, _ = sampler.sample(S=num_inference_steps,
                                                conditioning=c,
                                                batch_size=1,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=guidance_scale,
                                                unconditional_conditioning=uc,
                                                eta=0,
                                                x_T=None)

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            for x_sample in x_samples:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                image = Image.fromarray(x_sample.astype(np.uint8))
    
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}


if __name__ == "__main__":
    init()
    inference({
    "prompt": "A monkey riding an elephant",
    "height": 768,
    "width": 768
    })