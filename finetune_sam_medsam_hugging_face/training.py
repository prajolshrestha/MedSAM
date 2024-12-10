from datasets import Dataset, Image as DatasetsImage
import transformers

import torch
from torch.utils.data import Dataset as torchDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn.functional import threshold, normalize
from torch.nn.functional import interpolate
from torch.optim import Adam
from transformers import SamModel, SamProcessor

import numpy as np
import glob
from PIL import Image
from tifffile import imread
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import os


import monai
from tqdm import tqdm
from statistics import mean





def create_dataset(images, labels):
    dataset = Dataset.from_dict({"image": images, 
                                 "label": labels})
    dataset.cast_column("image", DatasetsImage())
    dataset.cast_column("label", DatasetsImage())

    return dataset


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def get_bounding_box(ground_truth_map):
    # Find all unique labels, excluding the background (usually 0)
    unique_labels = np.unique(ground_truth_map)
    unique_labels = unique_labels[unique_labels != 0]
    
    bounding_boxes = []
    
    for label in unique_labels:
        # Create a binary mask for this label
        binary_mask = (ground_truth_map == label)
        
        # Find properties of connected regions
        props = measure.regionprops(binary_mask.astype(int))
        
        if props:
            # Get bounding box
            minr, minc, maxr, maxc = props[0].bbox
            
            # Add perturbation to bounding box coordinates
            H, W = ground_truth_map.shape
            minc = max(0, minc - np.random.randint(0, 20))
            maxc = min(W, maxc + np.random.randint(0, 20))
            minr = max(0, minr - np.random.randint(0, 20))
            maxr = min(H, maxr + np.random.randint(0, 20))
            
            # Append bounding box in the format [x_min, y_min, x_max, y_max]
            bounding_boxes.append([minc, minr, maxc, maxr])
    
    return bounding_boxes



class SAMDataset(torchDataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = np.array(item["image"])
    # Ensure image is 3D (H, W, C)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.repeat(image, 3, axis=-1)  # Repeat to get 3 channels
    elif len(image.shape) == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)  # Repeat to get 3 channels
        

    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs
  


def collate_fn(batch):
    # Separate different items in the batch
    pixel_values = [item['pixel_values'] for item in batch]
    input_boxes = [item['input_boxes'] for item in batch]
    ground_truth_masks = [torch.from_numpy(item['ground_truth_mask']) for item in batch]

    # Pad and stack pixel_values
    pixel_values = torch.stack(pixel_values)

    # Pad input_boxes to the same size
    max_boxes = max(boxes.size(0) for boxes in input_boxes)
    input_boxes_padded = torch.stack([
        torch.nn.functional.pad(boxes, (0, 0, 0, max_boxes - boxes.size(0)))
        for boxes in input_boxes
    ])

    # Pad ground truth masks to the same size
    max_h = max(mask.size(0) for mask in ground_truth_masks)
    max_w = max(mask.size(1) for mask in ground_truth_masks)
    ground_truth_masks_padded = torch.stack([
        torch.nn.functional.pad(mask, (0, max_w - mask.size(1), 0, max_h - mask.size(0)))
        for mask in ground_truth_masks
    ])

    return {
        'pixel_values': pixel_values,
        'input_boxes': input_boxes_padded,
        'ground_truth_mask': ground_truth_masks_padded
    }



image_folder = "/home/hpc/iwi5/iwi5171h/data_processing/calculated_dataset/stardist_dataset/closest_vertex_dataset/closest_vertex_high_degree_only_120/train/images/"
mask_folder = "/home/hpc/iwi5/iwi5171h/data_processing/calculated_dataset/stardist_dataset/closest_vertex_dataset/closest_vertex_high_degree_only_120/train/masks/"

image_filename = sorted(glob.glob(image_folder + "*.tif"))
mask_filename = sorted(glob.glob(mask_folder + "*.tif"))

# Load images and masks
images = [io.imread(img_path) for img_path in image_filename]
labels = [io.imread(mask_path) for mask_path in mask_filename]

# Create the dataset
dataset = create_dataset(images, labels)


device = "cuda" if torch.cuda.is_available() else "cpu"

# MedSAM
# processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
# model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(device)

# SAM
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base")

train_dataset = SAMDataset(dataset=dataset, processor=processor)


# Update the DataLoader with the new collate_fn
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)




# Note: Hyperparameter tuning could improve performance here
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean', batch=True)

###Training start

num_epochs = 100
best_loss = float('inf')
#save_path = 'best_model.pth'
save_path = 'best_mask_decoder.pth'

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=True)

      # compute loss
      predicted_masks = outputs.pred_masks #.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      # Resize ground truth masks to match predicted masks
      ground_truth_masks = interpolate(ground_truth_masks.unsqueeze(1), 
                                        size=predicted_masks.shape[-2:], 
                                        mode='nearest')  # Shape: [B, 1, H, W]
      
      
      # Ensure both tensors have the same batch size
      batch_size = min(predicted_masks.shape[0], ground_truth_masks.shape[0])
      #num_masks = predicted_masks.shape[1]
      predicted_masks = predicted_masks[:batch_size]
      ground_truth_masks = ground_truth_masks[:batch_size]

      # Repeat ground truth masks to match the number of predicted masks
      ground_truth_masks = ground_truth_masks.repeat(1, predicted_masks.shape[1], 1, 1, 1)


      # Flatten the mask dimension into the batch dimension
      predicted_masks = predicted_masks.reshape(-1, 1, *predicted_masks.shape[-2:])
      ground_truth_masks = ground_truth_masks.reshape(-1, 1, *ground_truth_masks.shape[-2:])
  
      #Ensure both tensors have the same number of elements in the first dimension
      min_masks = min(predicted_masks.shape[0], ground_truth_masks.shape[0])
      predicted_masks = predicted_masks[:min_masks]
      ground_truth_masks = ground_truth_masks[:min_masks]

      #loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
      loss = seg_loss(predicted_masks, ground_truth_masks)

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    mean_loss = mean(epoch_losses)
    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean_loss}')

 
    # Save the model if it's the best so far
    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save({
            'epoch': epoch,
            'mask_decoder_state_dict': model.mask_decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, save_path)
        print(f'Mask decoder saved to {save_path}')


print("Training completed. Best model saved.")

