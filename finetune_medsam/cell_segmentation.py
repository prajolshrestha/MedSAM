import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from skimage import measure
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Set seeds for reproducibility
torch.manual_seed(2023)
torch.cuda.empty_cache()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

class CellDataset(Dataset):
    def __init__(self, data_root, bbox_padding=0, image_size=1024):
        self.data_root = data_root
        self.mask_path = os.path.join(data_root, "masks")
        self.img_path = os.path.join(data_root, "images")
        self.mask_files = sorted(glob.glob(os.path.join(self.mask_path, "*.tif")))  # Keep .tif
        self.mask_files = [
            file for file in self.mask_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_padding = bbox_padding
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        print(f"Number of images: {len(self.mask_files)}")

    def __len__(self):
        return len(self.mask_files)

    def extract_bboxes(self, mask):
        bboxes = []
        for label in np.unique(mask):
            if label == 0:  # Skip background
                continue
            binary_mask = (mask == label).astype(np.uint8)
            props = measure.regionprops(binary_mask)
            if props:
                bbox = props[0].bbox # (min_row, min_col, max_row, max_col)
                bboxes.append(self.add_padding(bbox))
        return bboxes

    def add_padding(self, bbox):
        y1, x1, y2, x2 = bbox
        #x1, y1, x2, y2 = bbox
        return [
            max(0, x1 - self.bbox_padding),
            max(0, y1 - self.bbox_padding),
            min(self.image_size, x2 + self.bbox_padding),
            min(self.image_size, y2 + self.bbox_padding)
        ]#add padding and resolve boundary 

    def __getitem__(self, index):
        img_name = os.path.basename(self.mask_files[index])
        img_path = os.path.join(self.img_path, img_name)
        mask_path = self.mask_files[index]

        # Read .tif images and rename to .tiff when opening?
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        # Convert image to RGB and resize
        img = img.convert('RGB')
        img = self.transform(img)# tensor

        # Resize mask
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = np.array(mask)

        # Extract bounding boxes from the multi-color mask
        bboxes = self.extract_bboxes(mask)

        # Normalize bounding box coordinates
        bboxes = torch.tensor(bboxes).float() # list to tensor
        bboxes[:, [0, 2]] /= self.image_size # normalize
        bboxes[:, [1, 3]] /= self.image_size

        # Convert multi-instance mask to one-hot encoding
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background
        one_hot_mask = np.zeros((len(unique_labels), self.image_size, self.image_size), dtype=np.uint8)
        
        for i, label in enumerate(unique_labels):
            one_hot_mask[i] = (mask == label).astype(np.uint8)

        return (
            img,
            torch.tensor(one_hot_mask).long(),
            bboxes,
            img_name,
        )

class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        
        batch_size = image.shape[0]
        outputs = []
        
        for i in range(batch_size):
            image_embed = image_embedding[i].unsqueeze(0)
            box_batch = boxes[i]
            
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=box_batch.unsqueeze(1),  # (N, 1, 4)
                    masks=None,
                )
            
            low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            ori_res_masks = F.interpolate(
                low_res_masks,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            outputs.append(ori_res_masks.squeeze(1))  # Remove the extra dimension
        
        return torch.stack(outputs, dim=0)  # Stack along the batch dimension

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_path", type=str, default="/home/hpc/iwi5/iwi5171h/data_processing/calculated_dataset/stardist_dataset/closest_vertex_dataset/closest_vertex_high_degree_only_120/train",
                        help="path to cell image data; two subfolders: masks and images")
    parser.add_argument("-task_name", type=str, default="CellSAM")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("-checkpoint", type=str, default="/home/hpc/iwi5/iwi5171h/medsam_file/MedSAM/work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("-num_epochs", type=int, default=10)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-lr", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("-weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device(args.device)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join("work_dir", args.task_name + "-" + run_id)
    os.makedirs(model_save_path, exist_ok=True)

    # Check if the data path exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {args.data_path}")

    # Initialize dataset
    train_dataset = CellDataset(args.data_path)
    
    # Check if the dataset is empty
    if len(train_dataset) == 0:
        raise ValueError(f"The dataset is empty. Please check the data path: {args.data_path}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize MedSAM model
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    # Set up optimizer and loss functions
    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(medsam_model.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # Training loop
    num_epochs = args.num_epochs
    losses = []
    best_loss = 1e10

    for epoch in range(num_epochs):
        epoch_loss = 0
        for step, (image, mask, bboxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            image, mask = image.to(device), mask.to(device)
            bboxes = bboxes.to(device)

            medsam_pred = medsam_model(image, bboxes)
            
            # Adjust predictions to match the number of ground truth instances
            num_instances = mask.shape[1]
            if medsam_pred.shape[1] < num_instances:
                medsam_pred = F.pad(medsam_pred, (0, 0, 0, 0, 0, num_instances - medsam_pred.shape[1]))
            elif medsam_pred.shape[1] > num_instances:
                medsam_pred = medsam_pred[:, :num_instances]
            
            # Calculate loss
            loss = seg_loss(medsam_pred, mask) + ce_loss(medsam_pred, mask.float())
            
            loss.backward()
            optimizer.step()


            epoch_loss += loss.item()

        epoch_loss /= (step + 1)
        losses.append(epoch_loss)
        print(f'Epoch: {epoch}, Loss: {epoch_loss}')

        # Save the latest model
        torch.save(medsam_model.state_dict(), os.path.join(model_save_path, "medsam_model_latest.pth"))

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(medsam_model.state_dict(), os.path.join(model_save_path, "medsam_model_best.pth"))

    # Plot and save loss curve
    plt.plot(losses)
    plt.title("Dice + Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(model_save_path, args.task_name + "_train_loss.png"))
    plt.close()

if __name__ == "__main__":
    main()