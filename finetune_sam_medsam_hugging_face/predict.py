import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def predict_and_visualize(image_path, model, processor, device):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = np.array(image)
    
    # Prepare the image and prompt for the model
    inputs = processor(image,  return_tensors="pt")
    
    # Move inputs to the appropriate device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate masks
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    
    # Get the predicted masks
    masks = outputs.pred_masks.squeeze().cpu().numpy()
    
    # Visualize the results
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca())
    plt.axis('off')
    plt.show()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the base model
    model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
    print('Base model loaded!')
    
    # Load the trained mask decoder
    checkpoint = torch.load('best_mask_decoder.pth', map_location=device)
    model.mask_decoder.load_state_dict(checkpoint['mask_decoder_state_dict'])
    print('Finetuned model loaded!')
    
    # Set the model to evaluation mode
    model.eval()
    
    # Predict on a new image
    image_path = "/home/hpc/iwi5/iwi5171h/data_processing/calculated_dataset/stardist_dataset/closest_vertex_dataset/closest_vertex_high_degree_only_120/train/images/4571_od_129_7681_4_0_calculated_ref_51_lps_8_lbss_8_sr_n_50_cropped_5_0_0.tif"
    predict_and_visualize(image_path, model, processor, device)

if __name__ == "__main__":
    main()