import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from segment_anything import sam_model_registry
from skimage import io, transform, measure  # {{ edit_1 }}
import torch.nn.functional as F
import argparse
from PIL import Image
from cell_segmentation import MedSAM

def generate_random_boxes(image_size, num_boxes, excluded_boxes, min_size=10, max_size=18, max_attempts=1000):
    boxes = []
    H, W = image_size
    for _ in range(num_boxes):
        for _ in range(max_attempts):
            size_h = np.random.randint(min_size, min(max_size, H))
            size_w = np.random.randint(min_size, min(max_size, W))
            x1 = np.random.randint(0, W - size_w)
            y1 = np.random.randint(0, H - size_h)
            x2, y2 = x1 + size_w, y1 + size_h

            new_box = [x1, y1, x2, y2]
            if not any(overlap(new_box, box) for box in excluded_boxes + boxes):
                boxes.append(new_box)
                break
    return boxes

def overlap(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

def refine_box(mask, original_box, scale=1.2):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        return original_box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    box = [
        max(x_min - int(scale * (x_max - x_min) / 2), 0),
        max(y_min - int(scale * (y_max - y_min) / 2), 0),
        min(x_max + int(scale * (x_max - x_min) / 2), original_box[2]),
        min(y_max + int(scale * (y_max - y_min) / 2), original_box[3])
    ]
    return box

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

@torch.no_grad()
def cellsam_inference(cellsam_model, img_embed, boxes, H=1024, W=1024):
    boxes_torch = torch.as_tensor(boxes, dtype=torch.float, device=img_embed.device)

    sparse_embeddings, dense_embeddings = cellsam_model.prompt_encoder(
        points=None,
        boxes=boxes_torch,
        masks=None,
    )
    
    low_res_masks, _ = cellsam_model.mask_decoder(
        image_embeddings=img_embed.repeat(len(boxes), 1, 1, 1),
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
    parser = argparse.ArgumentParser(description="Run inference using CellSAM model with refined bounding boxes")
    parser.add_argument("-i", "--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("-o", "--output_path", type=str, default="./", help="Path to save the output")
    parser.add_argument("--initial_num_boxes", type=int, default=100, help="Initial number of random bounding boxes to generate")
    parser.add_argument("--num_boxes_per_iteration", type=int, default=100, help="Number of boxes to add per iteration")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    parser.add_argument("-chk", "--checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device(args.device)
    
    sam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    cellsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    cellsam_model.eval()

    img = Image.open(args.image_path)
    img_np = np.array(img)
    if len(img_np.shape) == 2:
        img_np = np.repeat(img_np[:, :, None], 3, axis=-1)
    H, W, _ = img_np.shape
    
    img_1024 = transform.resize(img_np, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    image_embedding = cellsam_model.image_encoder(img_1024_tensor)

    fixed_boxes = []    # {{ edit_2 }}
    all_masks = []      # {{ edit_3 }}
    combined_mask = np.zeros((H, W), dtype=bool)
    iteration = 0
    max_iterations = 5  # {{ edit_4 }}
    
    # Define bounding box size constraints
    min_box_size = 10  # {{ edit_5 }}
    max_box_size = 18  # {{ edit_6 }}

    while iteration < max_iterations:
        if iteration == 0:
            boxes = generate_random_boxes((H, W), args.initial_num_boxes, fixed_boxes)
        else:
            boxes = generate_random_boxes((H, W), args.num_boxes_per_iteration, fixed_boxes)
        if not boxes:
            break

        boxes_1024 = np.array(boxes) / np.array([W, H, W, H]) * 1024
        masks = cellsam_inference(cellsam_model, image_embedding, boxes_1024, H, W)

        for box, mask in zip(boxes, masks):
            mask_binary = mask[0] > 0.5
            combined_mask = combined_mask | mask_binary  # {{ edit_7 }}
            all_masks.append(mask)                        # {{ edit_8 }}
            fixed_boxes.append(box)                        # {{ edit_9 }}

        # Compute bounding boxes from the combined mask
        regions = measure.regionprops(combined_mask.astype(int))  # {{ edit_10 }}
        if not regions:
            break
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            box_w = maxc - minc
            box_h = maxr - minr

            # Reduce box size as much as possible while ensuring the cell is inside
            # No scaling factor; tighten the box to the exact mask
            new_box = [minc, minr, maxc, maxr]

            # Ensure box width and height are within constraints
            current_box_w = new_box[2] - new_box[0]
            current_box_h = new_box[3] - new_box[1]

            # Adjust box width
            if current_box_w < min_box_size:
                extra_w = (min_box_size - current_box_w) // 2
                new_minc = max(new_box[0] - extra_w, 0)
                new_maxc = min(new_box[2] + extra_w, W)
                # If still smaller after expansion, adjust accordingly
                if (new_maxc - new_minc) < min_box_size:
                    new_maxc = min(new_minc + min_box_size, W)
                new_box = [new_minc, new_box[1], new_maxc, new_box[3]]

            # Adjust box height
            if current_box_h < min_box_size:
                extra_h = (min_box_size - current_box_h) // 2
                new_miny = max(new_box[1] - extra_h, 0)
                new_maxy = min(new_box[3] + extra_h, H)
                # If still smaller after expansion, adjust accordingly
                if (new_maxy - new_miny) < min_box_size:
                    new_maxy = min(new_miny + min_box_size, H)
                new_box = [new_box[0], new_miny, new_box[2], new_maxy]

            # Ensure box does not exceed maximum size
            final_box_w = new_box[2] - new_box[0]
            final_box_h = new_box[3] - new_box[1]

            if final_box_w > max_box_size:
                center_x = (new_box[0] + new_box[2]) // 2
                half_w = max_box_size // 2
                new_box[0] = max(center_x - half_w, 0)
                new_box[2] = min(center_x + half_w, W)
                if (new_box[2] - new_box[0]) < max_box_size:
                    new_box[2] = min(new_box[0] + max_box_size, W)

            if final_box_h > max_box_size:
                center_y = (new_box[1] + new_box[3]) // 2
                half_h = max_box_size // 2
                new_box[1] = max(center_y - half_h, 0)
                new_box[3] = min(center_y + half_h, H)
                if (new_box[3] - new_box[1]) < max_box_size:
                    new_box[3] = min(new_box[1] + max_box_size, H)

            refined_box = [new_box[0], new_box[1], new_box[2], new_box[3]]
            fixed_boxes.append(refined_box)  # {{ edit_11 }}

        iteration += 1  # {{ edit_12 }}

    # Visualize and save results
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    
    num_masks = len(all_masks)
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_masks))
    
    combined_visual_mask = np.zeros((H, W, 4))
    for i, mask in enumerate(all_masks):
        mask_binary = mask[0] > 0.5
        color_mask = np.zeros((H, W, 4))
        color_mask[mask_binary] = colors[i]
        combined_visual_mask += color_mask * (1 - combined_visual_mask[:,:,3:])
    
    plt.imshow(combined_visual_mask)
    
    for box in fixed_boxes:
        show_box(box, plt.gca())
    
    plt.title("CellSAM Segmentation with Refined Bounding Boxes")
    plt.axis('off')
    output_filename = os.path.join(args.output_path, f"cellsam_output_refined_bbox_{os.path.splitext(os.path.basename(args.image_path))[0]}.png")
    plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
    print(f"Output saved to {output_filename}")

    # Optionally save individual masks
    # for i, mask in enumerate(all_masks):
    #     binary_mask = (mask[0] > 0.5).astype(np.uint8) * 255
    #     mask_filename = os.path.join(args.output_path, f"cellsam_mask_refined_bbox_{i}_{os.path.splitext(os.path.basename(args.image_path))[0]}.png")
    #     io.imsave(mask_filename, binary_mask, check_contrast=False)
    #     print(f"Mask {i} saved to {mask_filename}")

if __name__ == "__main__":
    main()