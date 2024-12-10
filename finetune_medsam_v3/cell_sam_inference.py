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

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

@torch.no_grad()
def cellsam_inference(cellsam_model, img_embed, boxes_1024=None, H=1024, W=1024):
    if boxes_1024 is not None:
        boxes_torch = torch.as_tensor(boxes_1024, dtype=torch.float, device=img_embed.device)
    else:
        boxes_torch = None

    sparse_embeddings, dense_embeddings = cellsam_model.prompt_encoder(
        points=None,
        boxes=boxes_torch,
        masks=None,
    )
    
    low_res_masks, _ = cellsam_model.mask_decoder(
        image_embeddings=img_embed.repeat(len(boxes_torch), 1, 1, 1),
        image_pe=cellsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    masks = F.interpolate(
        low_res_masks,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks.sigmoid().cpu().numpy()
    return masks

def main():
    parser = argparse.ArgumentParser(description="Run inference using CellSAM model")
    parser.add_argument("-i", "--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("-o", "--output_path", type=str, default="./", help="Path to save the output")
    parser.add_argument("--boxes", type=str, default=None, help="Bounding boxes in format '[[x1,y1,x2,y2],[x1,y1,x2,y2],...]'")
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

    # Process bounding boxes if provided
    if args.boxes:
        boxes_np = np.array(eval(args.boxes))
        boxes_1024 = boxes_np / np.array([W, H, W, H]) * 1024
    else:
        boxes_1024 = None

    # Run inference
    masks = cellsam_inference(cellsam_model, image_embedding, boxes_1024, H, W)

    # Visualize and save results
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    
    # Create a colormap for the masks
    num_masks = len(masks)
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_masks))
    
    # Overlay all masks on a single image
    combined_mask = np.zeros((H, W, 4))
    for i, mask in enumerate(masks):
        mask_binary = mask[0] > 0.5
        color_mask = np.zeros((H, W, 4))
        color_mask[mask_binary] = colors[i]
        combined_mask += color_mask * (1 - combined_mask[:,:,3:])
    
    plt.imshow(combined_mask)
    
    if args.boxes:
        for box in boxes_np:
            show_box(box, plt.gca())
    
    plt.title("CellSAM Segmentation")
    plt.axis('off')
    output_filename = os.path.join(args.output_path, f"cellsam_output_{os.path.splitext(os.path.basename(args.image_path))[0]}.png")
    plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
    print(f"Output saved to {output_filename}")

    # # Save individual binary masks as .png files (optional)
    # for i, mask in enumerate(masks):
    #     binary_mask = (mask[0] > 0.5).astype(np.uint8) * 255
    #     mask_filename = os.path.join(args.output_path, f"cellsam_mask_{i}_{os.path.splitext(os.path.basename(args.image_path))[0]}.png")
    #     io.imsave(mask_filename, binary_mask, check_contrast=False)
    #     print(f"Mask {i} saved to {mask_filename}")

if __name__ == "__main__":
    main()