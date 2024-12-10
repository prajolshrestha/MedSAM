import numpy as np
import matplotlib.pyplot as plt
from cell_segmentation import CellDataset
import torch
import os
import argparse

def visualize_sample(data_path, sample_idx=0, output_dir='visualization_output'):
    # Create the CellDataset object
    dataset = CellDataset(data_path)

    # Get a sample from the dataset
    image, mask, bboxes, img_name = dataset[sample_idx]

    # Convert tensors to numpy arrays for visualization
    image_np = image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    mask_np = mask.numpy()
    bboxes_np = bboxes.numpy()
    #print(bboxes_np)
    #print(mask_np.shape)

    # Denormalize bounding box coordinates
    image_size = image_np.shape[0]  # Assuming square image
    bboxes_denorm = bboxes_np.copy()
    bboxes_denorm[:, [0, 2]] *= image_size
    bboxes_denorm[:, [1, 3]] *= image_size

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a larger figure
    plt.figure(figsize=(24, 6))

    # Add a small margin to each subplot
    margin = 0.05

    # Visualize the image
    plt.subplot(1, 4, 1)
    plt.imshow(image_np)
    plt.title(f"Image: {img_name}")
    plt.axis('off')
    plt.gca().set_position([plt.gca().get_position().x0 - margin, 
                            plt.gca().get_position().y0 - margin, 
                            plt.gca().get_position().width + 2*margin, 
                            plt.gca().get_position().height + 2*margin])

    # Visualize the mask (all instances combined with different colors)
    plt.subplot(1, 4, 2)
    combined_mask = np.zeros((*mask_np.shape[1:], 3), dtype=np.float32)
    for i, instance_mask in enumerate(mask_np):
        color = np.random.rand(3)
        combined_mask += instance_mask[:, :, None] * color
    combined_mask = np.clip(combined_mask, 0, 1)
    plt.imshow(combined_mask)
    plt.title("Combined Mask (Multi-color)")
    plt.axis('off')
    plt.gca().set_position([plt.gca().get_position().x0 - margin, 
                            plt.gca().get_position().y0 - margin, 
                            plt.gca().get_position().width + 2*margin, 
                            plt.gca().get_position().height + 2*margin])

    # Visualize the image with bounding boxes
    plt.subplot(1, 4, 3)
    plt.imshow(image_np)
    for bbox in bboxes_denorm:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.title("Image with Bounding Boxes")
    plt.axis('off')
    plt.gca().set_position([plt.gca().get_position().x0 - margin, 
                            plt.gca().get_position().y0 - margin, 
                            plt.gca().get_position().width + 2*margin, 
                            plt.gca().get_position().height + 2*margin])

    # Visualize individual instance masks
    num_instances = mask.shape[0]
    instance_grid = np.zeros((*mask_np.shape[1:], 3), dtype=np.float32)
    for i, instance_mask in enumerate(mask_np):
        color = np.random.rand(3)
        instance_grid += instance_mask[:, :, None] * color
    instance_grid = np.clip(instance_grid, 0, 1)
    plt.subplot(1, 4, 4)
    plt.imshow(instance_grid)
    plt.title(f"Individual Instances ({num_instances})")
    plt.axis('off')
    plt.gca().set_position([plt.gca().get_position().x0 - margin, 
                            plt.gca().get_position().y0 - margin, 
                            plt.gca().get_position().width + 2*margin, 
                            plt.gca().get_position().height + 2*margin])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sample_{sample_idx}_overview.png'))
    plt.close()

    print(f"Visualization saved for sample {sample_idx}")
    print(f"Number of instances: {num_instances}")
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Bounding boxes shape: {bboxes.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a sample from the CellDataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of the sample to visualize")
    parser.add_argument("--output_dir", type=str, default="visualization_output", help="Directory to save the output images")
    args = parser.parse_args()

    visualize_sample(args.data_path, args.sample_idx, args.output_dir)