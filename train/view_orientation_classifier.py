import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import xml.etree.ElementTree as ET
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import random
import numpy as np

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)
    
class MammographyDataset(Dataset):
    def __init__(self, xml_file, img_dir, transform=None):
        self.root = ET.parse(xml_file).getroot()
        self.img_dir = img_dir
        self.transform = transform
        self.data = self.parse_xml()

    def parse_xml(self):
        data = []
        for image in self.root.findall(".//image"):
            filename = image.get('name')
            view = 'CC' if image.find(".//tag[@label='CC']") is not None else 'MLO'
            orientation = 'Right' if image.find(".//tag[@label='Right']") is not None else 'Left'
            data.append((filename, view, orientation))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, view, orientation = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        view_label = 0 if view == 'CC' else 1
        orientation_label = 0 if orientation == 'Right' else 1
        
        return image, view_label, orientation_label

class Res2NextLightningModule(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.res2next = timm.create_model('res2next50', pretrained=True)
        in_features = self.res2next.fc.in_features
        self.res2next.fc = nn.Identity()
        self.view_classifier = nn.Linear(in_features, num_classes)
        self.orientation_classifier = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc_view = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_acc_orientation = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc_view = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc_orientation = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc_view = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc_orientation = Accuracy(task="multiclass", num_classes=num_classes)
        
    def forward(self, x):
        features = self.res2next(x)
        view_output = self.view_classifier(features)
        orientation_output = self.orientation_classifier(features)
        return view_output, orientation_output

    def training_step(self, batch, batch_idx):
        images, view_labels, orientation_labels = batch
        view_output, orientation_output = self(images)
        view_loss = self.criterion(view_output, view_labels)
        orientation_loss = self.criterion(orientation_output, orientation_labels)
        loss = view_loss + orientation_loss
        
        # Calculate and log training accuracy
        self.train_acc_view(view_output, view_labels)
        self.train_acc_orientation(orientation_output, orientation_labels)
        self.log('train_acc_view', self.train_acc_view, on_step=False, on_epoch=True)
        self.log('train_acc_orientation', self.train_acc_orientation, on_step=False, on_epoch=True)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, view_labels, orientation_labels = batch
        view_output, orientation_output = self(images)
        view_loss = self.criterion(view_output, view_labels)
        orientation_loss = self.criterion(orientation_output, orientation_labels)
        loss = view_loss + orientation_loss
        
        # Calculate validation accuracy
        self.val_acc_view(view_output, view_labels)
        self.val_acc_orientation(orientation_output, orientation_labels)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc_view', self.val_acc_view, on_epoch=True, prog_bar=True)
        self.log('val_acc_orientation', self.val_acc_orientation, on_epoch=True, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        images, view_labels, orientation_labels = batch
        view_output, orientation_output = self(images)
        view_loss = self.criterion(view_output, view_labels)
        orientation_loss = self.criterion(orientation_output, orientation_labels)
        loss = view_loss + orientation_loss
        
        # Calculate test accuracy
        self.test_acc_view(view_output, view_labels)
        self.test_acc_orientation(orientation_output, orientation_labels)
        
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc_view', self.test_acc_view, on_epoch=True, prog_bar=True)
        self.log('test_acc_orientation', self.test_acc_orientation, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

class PrintCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {metrics['train_loss_epoch']:.4f}")
        print(f"  Train View Accuracy: {metrics['train_acc_view']:.4f}")
        print(f"  Train Orientation Accuracy: {metrics['train_acc_orientation']:.4f}")
        print(f"  Validation Loss: {metrics['val_loss']:.4f}")
        print(f"  Validation View Accuracy: {metrics['val_acc_view']:.4f}")
        print(f"  Validation Orientation Accuracy: {metrics['val_acc_orientation']:.4f}")

    def on_test_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        print("\nTest Results:")
        print(f"  Test Loss: {metrics['test_loss']:.4f}")
        print(f"  Test View Accuracy: {metrics['test_acc_view']:.4f}")
        print(f"  Test Orientation Accuracy: {metrics['test_acc_orientation']:.4f}")

class PlotCallback(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_acc_views = []
        self.val_acc_views = []
        self.train_acc_orientations = []
        self.val_acc_orientations = []
        self.epochs = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self.epochs.append(trainer.current_epoch)
        self.train_losses.append(metrics['train_loss_epoch'].item())
        self.val_losses.append(metrics['val_loss'].item())
        self.train_acc_views.append(metrics['train_acc_view'].item())
        self.val_acc_views.append(metrics['val_acc_view'].item())
        self.train_acc_orientations.append(metrics['train_acc_orientation'].item())
        self.val_acc_orientations.append(metrics['val_acc_orientation'].item())

        self.update_plot()

    def update_plot(self):
        plt.figure(figsize=(12, 10))

        # Plot accuracy
        plt.subplot(2, 1, 1)
        plt.plot(self.epochs, self.train_acc_views, label='Train View Acc')
        plt.plot(self.epochs, self.val_acc_views, label='Val View Acc')
        plt.plot(self.epochs, self.train_acc_orientations, label='Train Orientation Acc')
        plt.plot(self.epochs, self.val_acc_orientations, label='Val Orientation Acc')
        plt.title('Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(2, 1, 2)
        plt.plot(self.epochs, self.train_losses, label='Train Loss')
        plt.plot(self.epochs, self.val_losses, label='Val Loss')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_plots.png')
        plt.close()
                
class MammographyDataModule(pl.LightningDataModule):
    def __init__(self, xml_file, img_dir, batch_size=32):
        super().__init__()
        self.xml_file = xml_file
        self.img_dir = img_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        full_dataset = MammographyDataset(self.xml_file, self.img_dir, transform=transform)
        
        # Calculate sizes for splits
        dataset_size = len(full_dataset)
        train_size = int(0.7 * dataset_size)
        val_size = int(0.15 * dataset_size)
        test_size = dataset_size - train_size - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

def main():
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Set up data module
    data_module = MammographyDataModule(xml_file='annotations.xml', img_dir=r'D:\Code\dcm2png\png_data', batch_size=32)

    # Set up model
    model = Res2NextLightningModule()

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='res2next-mammography-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    print_callback = PrintCallback()
    plot_callback = PlotCallback()

    # Set up logger
    logger = TensorBoardLogger("lightning_logs", name="res2next_mammography_view_orientation")

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices='auto',
        callbacks=[checkpoint_callback, print_callback, plot_callback],  # Add plot_callback here
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=1
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, datamodule=data_module)

    # Load the best model
    best_model_path = checkpoint_callback.best_model_path
    best_model = Res2NextLightningModule.load_from_checkpoint(best_model_path)

    print(f"Best model saved at: {best_model_path}")

    # Test the best model
    trainer.test(best_model, datamodule=data_module)

if __name__ == "__main__":
    main()