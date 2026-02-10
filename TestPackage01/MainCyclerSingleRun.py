import os
import sys
import subprocess
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch
import numpy as np
from Flow_segmentation import Superimpose_image_with_masks, Predictor, Groundingdino_model, Segmentation_model
#Log ImportantInfo
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
scripts_dir = os.path.join(parent_dir, 'Scripts', 'libs')

sys.path.append(os.path.dirname(scripts_dir))

os.makedirs(scripts_dir, exist_ok=True)

parent_dir = os.path.dirname(os.getcwd())

scripts_dir = os.path.join(parent_dir, "Scripts")
sys.path.append(scripts_dir)


###### Change the text prompt here ######
text_prompt = 'Fingers and a hand'
print("Text input:", text_prompt)

#########################################

image_path = os.path.join(parent_dir, "Demo", "Fingers.png")
image_pil = Image.open(image_path).convert("RGB")

# plt.figure(figsize=(5, 5))
# plt.imshow(image_pil)
# plt.show()

image_pil = ImageOps.invert(image_pil)

######################### vit model type ##########################
sam_type="vit_h" # "vit_b" or "vit_l"

######### Auxiliary box in case text prompt does not work #########
aux_box = torch.tensor([[0, 200, 10, 10]])  # Replace with your auxiliary box in pixels x y w h

######### Point where you want to mask '1' or unmask '0' #########
input_point = None #np.array([[0, 100], [100, 450]]) # coordinates in pixels
input_label = None #np.array([0, 1]) # 1 for mask, 0 for unmask
###################################################################

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f"Device: {device}")

groundingdino_model = Groundingdino_model()
sam_model = Segmentation_model(sam_type, ckpt_path=None, device=device)

masks, boxes, phrases, logits = Predictor(image_pil, text_prompt, input_point, input_label, groundingdino_model, sam_model, aux_box, box_threshold=0.3, text_threshold=0.25)

image_pil = ImageOps.invert(image_pil) # if necessary

if len(masks) == 0:
    print(f"No objects of the '{text_prompt}' prompt detected in the image.")
else:
    # Convert masks to numpy arrays
    masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

    Superimpose_image_with_masks(image_pil, masks_np)
