import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
from torchvision.transforms import Resize
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
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
    image_np = np.array(image)

    # Check if the image is 2D (grayscale)
    if len(image_np.shape) == 2:
        # Convert 2D grayscale to 3D RGB
        image_np = np.stack((image_np,) * 3, axis=-1)
    
    # Prepare the image and prompt for the model
    inputs = processor(image_np, return_tensors="pt")
    
    # Move inputs to the appropriate device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate masks
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    
    # Apply sigmoid and convert to hard mask
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
    
    # Visualize the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image with predicted mask
    ax1.imshow(image)
    show_mask(medsam_seg, ax1)
    ax1.set_title("Predicted mask")
    ax1.axis('off')
    
    # Original image only
    ax2.imshow(image)
    ax2.set_title("Original image")
    ax2.axis('off')
    
    plt.tight_layout()
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