import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import MeanMetric
import logging
import os
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PectoralDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.data = self.load_data()
        logging.info(f"Loaded {len(self.data)} images from the dataset.")

    def load_data(self):
        data = []
        img_files = sorted(os.listdir(self.img_dir))
        mask_files = sorted(os.listdir(self.mask_dir))
        
        for img_file, mask_file in zip(img_files, mask_files):
            if img_file.split('.')[0] == mask_file.split('.')[0]:  # Ensure matching pairs
                data.append((img_file, mask_file))
            else:
                logging.warning(f"Mismatch between image {img_file} and mask {mask_file}")
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, mask_name = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            mask = Image.open(mask_path).convert('L')  # Ensure mask is also grayscale

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)

            # Ensure mask is binary
            mask = (mask > 0).float()

            return image, mask
        except Exception as e:
            logging.error(f"Error processing image {img_name} or mask {mask_name}: {str(e)}")
            return None

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.down5 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(1024, 1024))
        
        self.up1 = nn.ConvTranspose2d(1024, 1024, 2, stride=2)
        self.up_conv1 = DoubleConv(2048, 1024)
        self.up2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up_conv2 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv3 = DoubleConv(512, 256)
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv4 = DoubleConv(256, 128)
        self.up5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv5 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        x = self.up1(x6)
        x = torch.cat([x, x5], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x4], dim=1)
        x = self.up_conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv4(x)
        
        x = self.up5(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv5(x)
        
        logits = self.outc(x)
        return logits

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return 1 - dice

class PectoralSegmentation(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.model = UNet(n_channels=1, n_classes=1)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.learning_rate = learning_rate
        
        # Metrics
        self.train_dice = MeanMetric()
        self.val_dice = MeanMetric()

    def forward(self, x):
        return self.model(x)

    def calculate_dice_coefficient(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).float()
        intersection = (predictions * targets).sum()
        dice = (2. * intersection) / (predictions.sum() + targets.sum() + 1e-8)
        return dice

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        
        bce_loss = self.bce_loss(outputs, masks)
        dice_loss = self.dice_loss(outputs, masks)
        total_loss = (bce_loss + dice_loss) / 2

        dice_coeff = self.calculate_dice_coefficient(outputs, masks)
        self.train_dice.update(dice_coeff)

        self.log('train_bce_loss', bce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dice_loss', dice_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dice_coeff', dice_coeff, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        
        bce_loss = self.bce_loss(outputs, masks)
        dice_loss = self.dice_loss(outputs, masks)
        total_loss = (bce_loss + dice_loss) / 2

        dice_coeff = self.calculate_dice_coefficient(outputs, masks)
        self.val_dice.update(dice_coeff)

        self.log('val_bce_loss', bce_loss, on_epoch=True, prog_bar=True)
        self.log('val_dice_loss', dice_loss, on_epoch=True, prog_bar=True)
        self.log('val_total_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_dice_coeff', dice_coeff, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        self.log("train_dice", self.train_dice.compute(), on_epoch=True)
        self.train_dice.reset()

    def on_validation_epoch_end(self):
        self.log("val_dice", self.val_dice.compute(), on_epoch=True)
        self.val_dice.reset()

    def on_epoch_end(self):
        train_dice = self.trainer.callback_metrics.get("train_dice_coeff", None)
        val_dice = self.trainer.callback_metrics.get("val_dice_coeff", None)
        
        if train_dice is not None:
            print(f"Epoch {self.current_epoch} - Training Dice: {train_dice:.4f}")
        if val_dice is not None:
            print(f"Epoch {self.current_epoch} - Validation Dice: {val_dice:.4f}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_total_loss'
        }

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def create_data_loaders(dataset, batch_size, num_workers):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader

def main(args):
    logging.info("Starting the training process...")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    logging.info("Loading dataset...")
    dataset = PectoralDataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        transform=transform
    )

    logging.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    logging.info("Initializing model...")
    model = PectoralSegmentation(learning_rate=args.learning_rate)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_dice_coeff',
        dirpath=args.checkpoint_dir,
        filename=f'pectoral-segmentation-unet-{args.image_size}-{{epoch:02d}}-{{val_dice_coeff:.2f}}',
        save_top_k=3,
        mode='max',
        every_n_epochs=1
    )

    logger = TensorBoardLogger(args.log_dir, name=f"pectoral_segmentation_unet_{args.image_size}")

    logging.info("Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
    )

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        logging.info(f"Loading model from checkpoint: {args.checkpoint_path}")
        model = PectoralSegmentation.load_from_checkpoint(args.checkpoint_path)
    else:
        logging.info("Initializing new model")
        model = PectoralSegmentation(learning_rate=args.learning_rate)

    logging.info("Starting model training...")
    try:
        trainer.fit(model, train_loader, val_loader)
        logging.info("Training completed successfully!")
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
    
    logging.info("Training process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a pectoral muscle segmentation model")
    parser.add_argument("--img_dir", type=str, default="D:\\Code\\dcm2png\\png_data", help="Directory containing the image data")
    parser.add_argument("--mask_dir", type=str, default="D:\\Code\\dcm2png\\png_masks", help="Directory containing the mask data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--image_size", type=int, default=512, help="Size to resize images to")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="lightning_logs", help="Directory for TensorBoard logs")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator type (e.g., 'cpu', 'gpu', 'auto')")
    parser.add_argument("--devices", type=str, default="auto", help="Number of devices to use")
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--val_check_interval", type=float, default=1.0, help="Validation check interval")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a checkpoint file to resume training")
    
    args = parser.parse_args()
    main(args)