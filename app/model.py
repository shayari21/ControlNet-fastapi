#change working directory to parent directory of ControlNet
import sys
import os

# Get the parent directory of the current script (ControlNet/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
CONTROLNET_DIR = os.path.dirname(BASE_DIR)  

# Add ControlNet directory to sys.path
sys.path.append(CONTROLNET_DIR)

#import dependencies
import torch
from PIL import Image
import numpy as np
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything
import einops
import random
import config
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.image as mpimg
from utils import rgb2lab,lab2rgb,rgb2yuv,yuv2rgb,srgb2lin,lin2srgb,lin2srgb,take_luminance_from_first_chroma_from_second


class ControlNetModel:
    def __init__(self):
        # Load the pre-trained model 
        # Load Canny Edge Detector
        self.apply_canny = CannyDetector()

        # Define relative paths to the model files based on the current working directory
        # Change model for other modes. Current mode set for canny edge detection. 
        model_config_path = os.path.join(CONTROLNET_DIR, 'models', 'cldm_v15.yaml')
        model_weights_path = os.path.join(CONTROLNET_DIR,'models', 'control_sd15_canny.pth')

        # Load the model using the relative paths
        self.model = create_model(model_config_path).cpu()

        # Load the weights using the relative path
        self.model.load_state_dict(load_state_dict(model_weights_path, location='cuda'))
        self.model = self.model.cuda()
        
        # Initialize DDIM Sampler
        self.ddim_sampler = DDIMSampler(self.model)
    

    def generate_synthetic_image(self, input_image, prompt):
        # Generate synthetic image from the model, typically with input in the form of image and text prompt

        #setting parameters
        num_samples = 1
        image_resolution = 512
        strength = 1.0
        guess_mode = False
        low_threshold =  50
        high_threshold = 100
        ddim_steps = 10
        scale = 9.0
        seed = 1
        eta = 0.0

        #positive and negative default prompts
        a_prompt = 'good quality' # 'best quality, extremely detailed' 
        n_prompt = 'animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

        with torch.no_grad():
            #input image processing(transforms are optional)
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            #applies Canny edge detection to extract edges.
            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            #converts edge maps to torch tensor and loads it to cuda
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            #reproducibility by adding random seed
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)
            
            #determining output resolution and scales
            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            
            #using DDIM to generate images
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            #decoding output to images
            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            result = [x_samples[i] for i in range(num_samples)]
            index=-1 #last sample from results list

            #using luminance function from awesomedemo.py
            test = take_luminance_from_first_chroma_from_second(resize_image(HWC3(input_image), image_resolution), result[index], mode="lab")
            
            #combining output with canny edge map, final generated image and input image and returning it.
            input_image_PIL=Image.fromarray(input_image).convert("RGB")
            width, height = input_image_PIL.size
            output1=Image.fromarray(result[index]).convert("RGB")
            output2= Image.fromarray(255 - detected_map).convert("RGB")
            output_image1 = output1.resize((width, height))
            output_image2 = output2.resize((width, height))
            combined_width = width * 3
            combined_image = Image.new("RGB", (combined_width, height))
            combined_image.paste(input_image_PIL, (width * 2, 0))
            combined_image.paste(output_image1, (width, 0))
            combined_image.paste(output_image2, (0, 0))
        return combined_image

