#!/usr/bin/env python3
"""
🎸 Fortnite Jam Battle - Real-time Note Detection Model
=====================================================

Trains a lightweight CNN for 60fps inference on 5 ROIs simultaneously.
Optimized for RTX 3090 with extensive data augmentation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import torch.nn.functional as F

import cv2
import numpy as np
import os
import json
from pathlib import Path
import time
import random
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

class NoteDataset(Dataset):
    def __init__(self, data_folder, transform=None, is_training=True):
        self.data_folder = Path(data_folder)
        self.transform = transform
        self.is_training = is_training
        
        # Class mapping
        self.classes = ['note', 'line', 'liftoff', 'blank']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all images and labels
        self.samples = []
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {'training' if is_training else 'validation'}")
        self._print_class_distribution()
    
    def _load_samples(self):
        """Load all image paths and labels"""
        for class_name in self.classes:
            class_folder = self.data_folder / class_name
            if not class_folder.exists():
                continue
            
            for img_path in class_folder.glob("*.jpg"):
                self.samples.append({
                    'path': img_path,
                    'label': self.class_to_idx[class_name],
                    'class_name': class_name
                })
    
    def _print_class_distribution(self):
        """Print class distribution"""
        class_counts = {}
        for sample in self.samples:
            class_name = sample['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\nClass distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / len(self.samples)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        label = sample['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class LightweightCNN(nn.Module):
    """
    Lightweight CNN optimized for real-time inference
    Target: <2ms inference time per ROI on RTX 3090
    """
    def __init__(self, num_classes=4, input_size=(72, 133)):  # Max ROI size
        super(LightweightCNN, self).__init__()
        
        # Efficient feature extraction
        self.features = nn.Sequential(
            # Block 1: 72x133 -> 36x66
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 36x66 -> 18x33
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 18x33 -> 9x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4: 9x16 -> 4x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class GameAugmentations:
    """Custom augmentations specific to guitar hero gameplay"""
    
    @staticmethod
    def slight_shift(image, max_shift=3):
        """Slight positional shifts"""
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        return TF.affine(image, angle=0, translate=[shift_x, shift_y], 
                        scale=1.0, shear=0, fill=0)
    
    @staticmethod
    def opacity_variation(image, opacity_range=(0.8, 1.2)):
        """Simulate slight opacity changes"""
        factor = random.uniform(*opacity_range)
        return TF.adjust_brightness(image, factor)
    
    @staticmethod
    def color_jitter_subtle(image):
        """Subtle color variations for different note types"""
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        saturation = random.uniform(0.8, 1.2)
        hue = random.uniform(-0.1, 0.1)
        
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        image = TF.adjust_saturation(image, saturation)
        image = TF.adjust_hue(image, hue)
        return image

def get_transforms(is_training=True, input_size=(72, 133)):
    """Get data transforms with extensive augmentation for training"""
    
    if is_training:
        transform = transforms.Compose([
            transforms.Resize(input_size),
            
            # Game-specific augmentations
            transforms.Lambda(lambda x: GameAugmentations.slight_shift(x, max_shift=2)),
            transforms.Lambda(lambda x: GameAugmentations.opacity_variation(x, (0.85, 1.15))),
            transforms.Lambda(lambda x: GameAugmentations.color_jitter_subtle(x)),
            
            # Standard augmentations
            transforms.RandomRotation(degrees=2),  # Very slight rotation
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small translation
            
            # Normalization
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def train_model(data_folder="training_data_augmented", epochs=100, batch_size=64, learning_rate=0.001):
    """Train the note detection model"""
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data preparation
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Load dataset
    full_dataset = NoteDataset(data_folder, transform=train_transform)
    
    # Train/validation split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"\n📊 Dataset split:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Model setup
    model = LightweightCNN(num_classes=4).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model parameters:")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=10, verbose=True)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    print(f"\n🎯 Starting training for {epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': full_dataset.classes
            }, 'best_note_detector.pth')
        
        # Record metrics
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Best: {best_val_acc:.2f}%")
        
        # Early stopping check
        if epoch > 20 and val_acc < best_val_acc - 10:  # Stop if validation drops significantly
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training curves
    plot_training_curves(train_losses, val_accuracies)
    
    # Benchmark inference speed
    benchmark_model(model, device)
    
    return model

def plot_training_curves(train_losses, val_accuracies):
    """Plot training loss and validation accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Validation accuracy
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📈 Training curves saved as 'training_curves.png'")

def benchmark_model(model, device, num_samples=1000):
    """Benchmark inference speed for real-time performance"""
    model.eval()
    
    # Create dummy input (5 ROIs simultaneously)
    dummy_input = torch.randn(5, 3, 72, 133).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_samples):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_batch = (total_time / num_samples) * 1000  # Convert to ms
    fps = 1000 / avg_time_per_batch  # FPS for processing 5 ROIs
    
    print(f"\n⚡ Inference Benchmark (5 ROIs simultaneously):")
    print(f"Average time per batch: {avg_time_per_batch:.2f} ms")
    print(f"Theoretical max FPS: {fps:.1f}")
    print(f"Target 60 FPS: {'✅ ACHIEVED' if fps >= 60 else '❌ TOO SLOW'}")

def evaluate_model(model_path="best_note_detector.pth", data_folder="training_data_augmented"):
    """Evaluate the trained model and show detailed metrics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = LightweightCNN(num_classes=4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    class_names = checkpoint['class_names']
    
    # Load test data
    test_transform = get_transforms(is_training=False)
    test_dataset = NoteDataset(data_folder, transform=test_transform, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print classification report
    print("\n📊 Detailed Classification Report:")
    print("=" * 50)
    print(classification_report(all_labels, all_predictions, 
                              target_names=class_names, digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    print("🎸 Fortnite Jam Battle - Note Detection Training")
    print("=" * 50)
    
    # Train the model
    model = train_model(
        data_folder="training_data_augmented",
        epochs=100,
        batch_size=64,
        learning_rate=0.001
    )
    
    # Evaluate the model
    print("\n" + "=" * 50)
    evaluate_model()
    
    print("\n🎯 Training complete! Use 'inference.py' for real-time detection.") 