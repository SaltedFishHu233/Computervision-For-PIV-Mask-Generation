import os
import sys
from PIL import Image, ImageOps
import torch
import numpy as np
from Flow_segmentation import Superimpose_image_with_masks, Predictor, Groundingdino_model, Segmentation_model
from Save_masks import save_masks_to_file
import torch.multiprocessing as mp
from datetime import datetime
def RepeatingSect(infoin,groundingdino_model,sam_model):
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
    text_prompt = 'White Particle on Black Background'
    print("Text input:", text_prompt)
    input_path=os.path.join(current_dir, "input")
    output_path=os.path.join(current_dir, "output")
    image_path = os.path.join(input_path,infoin)
    image_pil = Image.open(image_path).convert("RGB")

    # plt.figure(figsize=(5, 5))
    # plt.imshow(image_pil)
    # plt.show()

    image_pil = ImageOps.invert(image_pil)

    ######### Auxiliary box in case text prompt does not work #########
    aux_box = torch.tensor([[0, 200, 10, 10]])  # Replace with your auxiliary box in pixels x y w h

    ######### Point where you want to mask '1' or unmask '0' #########
    input_point = None #np.array([[0, 100], [100, 450]]) # coordinates in pixels
    input_label = None #np.array([0, 1]) # 1 for mask, 0 for unmask
    ###################################################################

    print("Starting SAM")
    masks, boxes, phrases, logits = Predictor(image_pil, text_prompt, input_point, input_label, groundingdino_model, sam_model, aux_box, box_threshold=0.3, text_threshold=0.25)

    image_pil = ImageOps.invert(image_pil) # if necessary

    if len(masks) == 0:
        print(f"No objects of the '{text_prompt}' prompt detected in the image.")
    else:
        # Convert masks to numpy arrays
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

        #Superimpose_image_with_masks(image_pil, masks_np)

    # Get the directory and base name of the file
    save_dir = os.path.dirname(output_path)+'/Output'
    print(save_dir)
    save_name = os.path.splitext(os.path.basename(image_path))[0]
    format = 'png' # 'npy', 'mat', 'png', or 'txt'
    try:
        save_masks_to_file(masks, save_dir, save_name, format)
        return 1
    except:
        return 0


def NoteBookSAM(infoin,groundingdino_model,sam_model):
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
    text_prompt = 'White Particle on Black Background'
    print("Text input:", text_prompt)
    input_path=os.path.join(current_dir, "input")
    output_path=os.path.join(current_dir, "output")
    image_path = os.path.join(input_path,infoin)
    image_pil = Image.open(image_path).convert("RGB")

    # plt.figure(figsize=(5, 5))
    # plt.imshow(image_pil)
    # plt.show()

    image_pil = ImageOps.invert(image_pil)

    ######### Auxiliary box in case text prompt does not work #########
    aux_box = torch.tensor([[0, 200, 10, 10]])  # Replace with your auxiliary box in pixels x y w h

    ######### Point where you want to mask '1' or unmask '0' #########
    input_point = None #np.array([[0, 100], [100, 450]]) # coordinates in pixels
    input_label = None #np.array([0, 1]) # 1 for mask, 0 for unmask
    ###################################################################

    print("Starting SAM")
    masks, boxes, phrases, logits = Predictor(image_pil, text_prompt, input_point, input_label, groundingdino_model, sam_model, aux_box, box_threshold=0.3, text_threshold=0.25)

    image_pil = ImageOps.invert(image_pil) # if necessary
    if len(masks) == 0:
        print(f"No objects of the '{text_prompt}' prompt detected in the image.")
    else:
        # Convert masks to numpy arrays
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

        #Superimpose_image_with_masks(image_pil, masks_np)

    # Get the directory and base name of the file
    save_dir = os.path.dirname(output_path)+'/temp'
    print(save_dir)
    save_name = os.path.splitext(os.path.basename(image_path))[0]
    format = 'png' # 'npy', 'mat', 'png', or 'txt'
    try:
        save_masks_to_file(masks, save_dir, save_name, format)
        return save_dir
    except:
        return 0



if __name__=='__main__':
    #Log ImportantInfo
    Starttime=datetime.now()
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    scripts_dir = os.path.join(parent_dir, 'Scripts', 'libs')

    sys.path.append(os.path.dirname(scripts_dir))

    os.makedirs(scripts_dir, exist_ok=True)

    parent_dir = os.path.dirname(os.getcwd())

    scripts_dir = os.path.join(parent_dir, "Scripts")
    sys.path.append(scripts_dir)


    #Start Cycle

    #########################################

    input_path=os.path.join(current_dir, "input")

    ImageA=os.listdir(input_path)

    num_processes = 4
    processes = []

    #model setup
    device='cpu'
    sam_type="vit_b" # "vit_b" or "vit_l"

    groundingdino_model = Groundingdino_model()
    sam_model = Segmentation_model(sam_type, ckpt_path=None, device=device)

    # for infoin in ImageA:
    #     p = mp.Process(target=RepeatingSect, args=(infoin,groundingdino_model,sam_model,))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    Counter=0
    for infoin in ImageA:
        RES=RepeatingSect(infoin,groundingdino_model,sam_model)
        Counter+=1
        print(Counter)

    EndTime=datetime.now()
    print("ProcessingEnd, TimeElapsed=")
    print(EndTime-Starttime)

    file_name = "Log.txt"
    content = "\nProcessingEnd, TimeElapsed="+str(EndTime-Starttime)

    # Open the file in write mode ('w')
    with open(file_name, 'w') as file:
        file.write(content)
    # pool = mp.Pool(processes=4)
    # results = [pool.apply_async(RepeatingSect, args=(x,groundingdino_model,sam_model,)) for x in ImageA]
    # Condition = [p.get() for p in results]
    # print(Condition)

    # image_path = os.path.join(input_path,"Fingers.png")
    # image_pil = Image.open(image_path).convert("RGB")

    # # plt.figure(figsize=(5, 5))
    # # plt.imshow(image_pil)
    # # plt.show()

    # image_pil = ImageOps.invert(image_pil)

    # ######################### vit model type ##########################
    # sam_type="vit_h" # "vit_b" or "vit_l"

    # ######### Auxiliary box in case text prompt does not work #########
    # aux_box = torch.tensor([[0, 200, 10, 10]])  # Replace with your auxiliary box in pixels x y w h

    # ######### Point where you want to mask '1' or unmask '0' #########
    # input_point = None #np.array([[0, 100], [100, 450]]) # coordinates in pixels
    # input_label = None #np.array([0, 1]) # 1 for mask, 0 for unmask
    # ###################################################################

    # # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # print(f"Device: {device}")

    # groundingdino_model = Groundingdino_model()
    # sam_model = Segmentation_model(sam_type, ckpt_path=None, device=device)

    # masks, boxes, phrases, logits = Predictor(image_pil, text_prompt, input_point, input_label, groundingdino_model, sam_model, aux_box, box_threshold=0.3, text_threshold=0.25)

    # image_pil = ImageOps.invert(image_pil) # if necessary

    # if len(masks) == 0:
    #     print(f"No objects of the '{text_prompt}' prompt detected in the image.")
    # else:
    #     # Convert masks to numpy arrays
    #     masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

    #     Superimpose_image_with_masks(image_pil, masks_np)

    # # Get the directory and base name of the file
    # save_dir = os.path.dirname(image_path)
    # save_name = os.path.splitext(os.path.basename(image_path))[0]
    # format = 'mat' # 'npy', 'mat', 'png', or 'txt'

    # save_masks_to_file(masks, save_dir, save_name, format)