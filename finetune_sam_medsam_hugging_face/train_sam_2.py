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
from skimage import io, transform
from skimage.measure import label


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

def convert_mask_to_instances(mask):
    # Label connected components
    labeled_mask = label(mask)
    
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(labeled_mask)[1:]
    
    # Create instance masks
    instance_masks = np.array([(labeled_mask == label).astype(int) for label in unique_labels])
    
    return instance_masks

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

    #-------------------------------------

    #instance_masks = convert_mask_to_instances(ground_truth_mask)

    #----------------------------------------

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



###Training start

# Train Multiple instance in a single image
def train_model(model, dataloader, optimizer, seg_loss, num_epochs, device, save_path):
    best_loss = float('inf')

    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)
            
            #print("pred_masks shape:", outputs.pred_masks.shape) # torch.Size([1, 152, 1, 256, 256])
            #print("iou_scores shape:", outputs.iou_scores.shape) # torch.Size([1, 152, 1])
            

            # compute loss
            predicted_masks = outputs.pred_masks # torch.Size([1, 152, 1, 256, 256])
            ground_truth_masks = batch["ground_truth_mask"].float().to(device) # torch.Size([1, 256, 256])

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ground_truth_masks = ground_truth_masks.unsqueeze(1).unsqueeze(1)
            # Shape: [2, 1, 1, 256, 256] (batch_size, 1, 1, height, width)

            # Repeat ground_truth_masks to match the number of instances in predicted_masks
            ground_truth_masks = ground_truth_masks.repeat(1, predicted_masks.shape[1], 1, 1, 1)
            # Shape: [2, 152, 1, 256, 256] (batch_size, num_instance, 1, height, width)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 

            ##########################################
            #print(f'predicted_masks shape: {predicted_masks.shape}')
            #print(f'Ground truth masks shape: {ground_truth_masks.shape}')
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


# Train single instance
#def train_model(model, dataloader, optimizer, seg_loss, num_epochs, device, save_path):
    # best_loss = float('inf')
    # model.to(device)

    # model.train()
    # for epoch in range(num_epochs):
    #     epoch_losses = []
    #     for batch in tqdm(dataloader):
    #         # forward pass
    #         outputs = model(pixel_values=batch["pixel_values"].to(device),
    #                         input_boxes=batch["input_boxes"].to(device),
    #                         multimask_output=False)

    #         # compute loss
    #         predicted_masks = outputs.pred_masks.squeeze(1)
    #         ground_truth_masks = batch["ground_truth_mask"].float().to(device)
    #         loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

    #         # backward pass (compute gradients of parameters w.r.t. loss)
    #         optimizer.zero_grad()
    #         loss.backward()

    #         # optimize
    #         optimizer.step()
    #         epoch_losses.append(loss.item())

    #     print(f'EPOCH: {epoch}')
    #     print(f'Mean loss: {mean(epoch_losses)}')


def main():
    image_folder = "/home/hpc/iwi5/iwi5171h/data_processing/calculated_dataset/stardist_dataset/closest_vertex_dataset/closest_vertex_high_degree_only_120/train/images/"
    mask_folder = "/home/hpc/iwi5/iwi5171h/data_processing/calculated_dataset/stardist_dataset/closest_vertex_dataset/closest_vertex_high_degree_only_120/train/masks/"

    image_filename = sorted(glob.glob(image_folder + "*.tif"))
    mask_filename = sorted(glob.glob(mask_folder + "*.tif"))

    # Load images and masks and resize it
    def resize_image(image, target_size=(256, 256)):
        return transform.resize(image, target_size, preserve_range=True, anti_aliasing=True).astype(image.dtype)

    def resize_mask(mask, target_size=(256, 256)):
        return transform.resize(mask, target_size, order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)

    images = [resize_image(io.imread(img_path)) for img_path in image_filename]
    labels = [resize_mask(io.imread(mask_path)) for mask_path in mask_filename]

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
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Hyperparameters
    num_epochs = 100
    learning_rate = 1e-5
    weight_decay = 0
    save_path = 'best_mask_decoder.pth'

    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(model.mask_decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean') #, batch=True)

    # Train the model
    train_model(model=model, 
                dataloader=train_dataloader, 
                optimizer=optimizer, 
                seg_loss=seg_loss, 
                num_epochs=num_epochs, 
                device=device, 
                save_path=save_path)
    

if __name__ == "__main__":
    main()