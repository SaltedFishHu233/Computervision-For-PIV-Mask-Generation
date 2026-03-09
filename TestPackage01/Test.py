import os

import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

import torch


def SAM3Ident(ImgDir,test_prompt='',save_dir=''):
    # turn on tfloat32 for Ampere GPUs
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load the model
    model = build_sam3_image_model()
    processor = Sam3Processor(model,confidence_threshold=0.1)

    # Load an image
    image = Image.open(ImgDir).convert('RGB')
    inference_state = processor.set_image(image)

    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt="black")

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    if len(masks) == 0:
        print(f"No objects of the prompt detected in the image.")
    else:
        # Convert masks to numpy arrays
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

        #Superimpose_image_with_masks(image_pil, masks_np)
        # Get the directory and base name of the file
        print(save_dir)
        # save_name = os.path.splitext(os.path.basename(image_path))[0]
        save_name='test'
        format = 'png' # 'npy', 'mat', 'png', or 'txt'
        print(type(masks_np))

        i=0
        # plt.show(masks_np[0])
        for image in masks_np:
            file_path_i = os.path.join(save_dir, f'{save_name}_mask_{i}.{format}')
            plt.imsave(file_path_i, image, cmap='gray')
            i+=1
