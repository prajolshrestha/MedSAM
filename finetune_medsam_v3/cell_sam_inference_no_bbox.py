import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse
from PIL import Image
from cell_segmentation import MedSAM

def show_mask(mask, ax, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

@torch.no_grad()
def cellsam_inference(cellsam_model, img_embed, H=1024, W=1024):
    # Use null prompts for the entire image
    sparse_embeddings, dense_embeddings = cellsam_model.prompt_encoder(
        points=None,
        boxes=None,
        masks=None,
    )
    
    low_res_masks, _ = cellsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=cellsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,  # Set to True to get multiple masks
    )
    print(low_res_masks.shape)

    masks = F.interpolate(
        low_res_masks,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks.sigmoid().cpu().numpy()
    print(masks.shape)
    return masks[0]  # Return all predicted masks

def main():
    parser = argparse.ArgumentParser(description="Run inference using CellSAM model without bounding boxes")
    parser.add_argument("-i", "--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("-o", "--output_path", type=str, default="./", help="Path to save the output")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    parser.add_argument("-chk", "--checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    device = torch.device(args.device)
    
    # Load the model
    sam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    cellsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    cellsam_model.eval()

    # Load and preprocess the image
    img = Image.open(args.image_path)
    img_np = np.array(img)
    if len(img_np.shape) == 2:
        img_np = np.repeat(img_np[:, :, None], 3, axis=-1)
    H, W, _ = img_np.shape
    
    img_1024 = transform.resize(img_np, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # Get image embedding
    with torch.no_grad():
        image_embedding = cellsam_model.image_encoder(img_1024_tensor)

    # Run inference
    masks = cellsam_inference(cellsam_model, image_embedding, H, W)
    print(masks.shape)

    # Visualize and save results
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    
    # Create a colormap for the masks
    num_masks = len(masks)
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_masks))
    
    # Overlay all masks on a single image
    combined_mask = np.zeros((H, W, 4))
    for i, mask in enumerate(masks):
        mask_binary = mask > 0.5
        color_mask = np.zeros((H, W, 4))
        color_mask[mask_binary] = colors[i]
        combined_mask += color_mask * (1 - combined_mask[:,:,3:])
    
    plt.imshow(combined_mask)
    
    plt.title("CellSAM Segmentation (No Bounding Boxes)")
    plt.axis('off')
    output_filename = os.path.join(args.output_path, f"cellsam_output_no_bbox_{os.path.splitext(os.path.basename(args.image_path))[0]}.png")
    plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
    print(f"Output saved to {output_filename}")

    # Save individual binary masks as .png files (optional)
    for i, mask in enumerate(masks):
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        mask_filename = os.path.join(args.output_path, f"cellsam_mask_no_bbox_{i}_{os.path.splitext(os.path.basename(args.image_path))[0]}.png")
        io.imsave(mask_filename, binary_mask, check_contrast=False)
        print(f"Mask {i} saved to {mask_filename}")

if __name__ == "__main__":
    main()