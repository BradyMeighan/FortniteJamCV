#!/usr/bin/env python3
"""
🎸 Dataset Augmentation Tool
============================

Generates multiple augmented variations of your classified samples to create
a much larger training dataset. This is crucial since 200 samples is too small!

Target: Generate 5000-10000+ samples from your existing classifications.
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance
import json
from tqdm import tqdm

class DatasetAugmenter:
    def __init__(self, input_folder="training_data_classified", output_folder="training_data_augmented"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.classes = ['note', 'line', 'liftoff', 'blank']
        
        # Augmentation parameters
        self.augmentations_per_sample = 40  # Generate 40 variations per original sample
        
        print(f"🔄 Dataset Augmenter initialized")
        print(f"Input: {self.input_folder}")
        print(f"Output: {self.output_folder}")
        print(f"Augmentations per sample: {self.augmentations_per_sample}")
    
    def setup_output_folders(self):
        """Create output folder structure"""
        for class_name in self.classes:
            output_class_folder = self.output_folder / class_name
            output_class_folder.mkdir(parents=True, exist_ok=True)
        
        # Copy originals folder for reference
        originals_folder = self.output_folder / "originals"
        originals_folder.mkdir(parents=True, exist_ok=True)
        for class_name in self.classes:
            (originals_folder / class_name).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Created output folders in {self.output_folder}")
    
    def shift_image(self, image, max_shift=3):
        """Shift image by small amounts (simulate slight misalignment)"""
        h, w = image.shape[:2]
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        
        # Create transformation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # Apply shift with border reflection to avoid black borders
        shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return shifted
    
    def rotate_image(self, image, max_angle=3):
        """Slightly rotate image"""
        h, w = image.shape[:2]
        angle = random.uniform(-max_angle, max_angle)
        
        # Get rotation matrix
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def scale_image(self, image, scale_range=(0.95, 1.05)):
        """Slightly scale image"""
        h, w = image.shape[:2]
        scale = random.uniform(*scale_range)
        
        # New dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if scale > 1.0:
            # Crop from center
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            result = scaled[start_y:start_y + h, start_x:start_x + w]
        else:
            # Pad to original size
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            result = cv2.copyMakeBorder(scaled, pad_y, h - new_h - pad_y, 
                                      pad_x, w - new_w - pad_x, cv2.BORDER_REFLECT)
        
        return result
    
    def adjust_brightness(self, image, brightness_range=(0.8, 1.2)):
        """Adjust brightness"""
        factor = random.uniform(*brightness_range)
        # Convert to float, apply factor, clip to valid range
        bright = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return bright
    
    def adjust_contrast(self, image, contrast_range=(0.8, 1.2)):
        """Adjust contrast"""
        factor = random.uniform(*contrast_range)
        # Convert to float, apply contrast, clip
        contrast = np.clip(128 + factor * (image.astype(np.float32) - 128), 0, 255).astype(np.uint8)
        return contrast
    
    def add_noise(self, image, noise_level=10):
        """Add slight gaussian noise"""
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    
    def adjust_saturation(self, image, saturation_range=(0.7, 1.3)):
        """Adjust color saturation"""
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust saturation
        factor = random.uniform(*saturation_range)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result
    
    def adjust_hue(self, image, hue_shift_range=(-10, 10)):
        """Slightly shift hue"""
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Shift hue
        shift = random.uniform(*hue_shift_range)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result
    
    def apply_blur(self, image, blur_range=(0, 1)):
        """Apply slight blur"""
        blur_amount = random.uniform(*blur_range)
        if blur_amount > 0:
            kernel_size = int(blur_amount * 2) * 2 + 1  # Ensure odd kernel size
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), blur_amount)
            return blurred
        return image
    
    def elastic_transform(self, image, alpha=5, sigma=2):
        """Apply elastic deformation (subtle)"""
        h, w = image.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.uniform(-alpha, alpha, (h, w)).astype(np.float32)
        dy = np.random.uniform(-alpha, alpha, (h, w)).astype(np.float32)
        
        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Create coordinate maps
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Apply transformation
        transformed = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return transformed
    
    def augment_sample(self, image, augmentation_type="random"):
        """Apply a combination of augmentations"""
        result = image.copy()
        
        # Geometric transformations (apply some randomly)
        if random.random() < 0.7:  # 70% chance
            result = self.shift_image(result, max_shift=random.randint(1, 4))
        
        if random.random() < 0.5:  # 50% chance
            result = self.rotate_image(result, max_angle=random.uniform(1, 3))
        
        if random.random() < 0.4:  # 40% chance
            result = self.scale_image(result, scale_range=(0.92, 1.08))
        
        # Color/lighting adjustments (apply some randomly)
        if random.random() < 0.8:  # 80% chance
            result = self.adjust_brightness(result, brightness_range=(0.7, 1.3))
        
        if random.random() < 0.6:  # 60% chance
            result = self.adjust_contrast(result, contrast_range=(0.7, 1.3))
        
        if random.random() < 0.5:  # 50% chance
            result = self.adjust_saturation(result, saturation_range=(0.6, 1.4))
        
        if random.random() < 0.3:  # 30% chance
            result = self.adjust_hue(result, hue_shift_range=(-15, 15))
        
        # Noise and blur (apply sparingly)
        if random.random() < 0.3:  # 30% chance
            result = self.add_noise(result, noise_level=random.uniform(5, 15))
        
        if random.random() < 0.2:  # 20% chance
            result = self.apply_blur(result, blur_range=(0.5, 1.5))
        
        if random.random() < 0.1:  # 10% chance (very subtle)
            result = self.elastic_transform(result, alpha=3, sigma=1)
        
        return result
    
    def process_class(self, class_name):
        """Process all samples in a class"""
        input_class_folder = self.input_folder / class_name
        output_class_folder = self.output_folder / class_name
        originals_class_folder = self.output_folder / "originals" / class_name
        
        if not input_class_folder.exists():
            print(f"⚠️  No {class_name} folder found, skipping...")
            return 0, 0
        
        # Get all original images
        image_files = list(input_class_folder.glob("*.jpg"))
        
        if not image_files:
            print(f"⚠️  No images found in {class_name} folder, skipping...")
            return 0, 0
        
        print(f"\n🔄 Processing {class_name} class...")
        print(f"Found {len(image_files)} original images")
        
        total_generated = 0
        
        # Process each original image
        for img_file in tqdm(image_files, desc=f"Augmenting {class_name}"):
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"⚠️  Could not load {img_file}")
                continue
            
            # Copy original to originals folder
            original_name = img_file.name
            cv2.imwrite(str(originals_class_folder / original_name), image)
            
            # Generate augmented versions
            base_name = img_file.stem
            
            for aug_idx in range(self.augmentations_per_sample):
                # Generate augmented image
                augmented = self.augment_sample(image)
                
                # Save with descriptive filename
                aug_filename = f"{base_name}_aug_{aug_idx:03d}.jpg"
                output_path = output_class_folder / aug_filename
                
                cv2.imwrite(str(output_path), augmented)
                total_generated += 1
        
        print(f"✅ Generated {total_generated} augmented samples for {class_name}")
        return len(image_files), total_generated
    
    def generate_balanced_dataset(self, target_samples_per_class=2000):
        """Generate a balanced dataset with target number of samples per class"""
        print(f"\n🎯 Generating balanced dataset ({target_samples_per_class} samples per class)")
        
        # First, check existing samples
        class_counts = {}
        for class_name in self.classes:
            input_class_folder = self.input_folder / class_name
            if input_class_folder.exists():
                class_counts[class_name] = len(list(input_class_folder.glob("*.jpg")))
            else:
                class_counts[class_name] = 0
        
        print(f"Current class distribution: {class_counts}")
        
        # Calculate how many augmentations needed per class
        for class_name in self.classes:
            original_count = class_counts[class_name]
            if original_count == 0:
                print(f"⚠️  No samples for {class_name}, skipping...")
                continue
            
            needed_total = target_samples_per_class
            augmentations_needed = max(0, needed_total - original_count)
            self.augmentations_per_sample = max(1, augmentations_needed // original_count)
            
            print(f"{class_name}: {original_count} originals -> generating {self.augmentations_per_sample} per sample")
            
            self.process_class(class_name)
    
    def run_augmentation(self, balanced=True, target_per_class=2000):
        """Run the full augmentation process"""
        print("🚀 Starting dataset augmentation...")
        
        # Setup output folders
        self.setup_output_folders()
        
        total_originals = 0
        total_generated = 0
        
        if balanced:
            self.generate_balanced_dataset(target_per_class)
        else:
            # Process each class
            for class_name in self.classes:
                originals, generated = self.process_class(class_name)
                total_originals += originals
                total_generated += generated
        
        # Final summary
        print(f"\n🎉 Augmentation complete!")
        print(f"Original samples: {total_originals}")
        print(f"Generated samples: {total_generated}")
        print(f"Total dataset size: {total_originals + total_generated}")
        
        # Check final distribution
        self.print_final_stats()
        
        # Save augmentation log
        self.save_augmentation_log(total_originals, total_generated)
    
    def print_final_stats(self):
        """Print final dataset statistics"""
        print(f"\n📊 Final Dataset Statistics:")
        print("-" * 50)
        
        total_samples = 0
        for class_name in self.classes:
            output_class_folder = self.output_folder / class_name
            if output_class_folder.exists():
                count = len(list(output_class_folder.glob("*.jpg")))
                total_samples += count
                print(f"{class_name}: {count} samples")
        
        print(f"Total: {total_samples} samples")
        
        if total_samples >= 5000:
            print("✅ Excellent! You now have a substantial dataset for training.")
        elif total_samples >= 2000:
            print("✅ Good! This should be sufficient for training.")
        else:
            print("⚠️  Still relatively small. Consider collecting more original samples.")
    
    def save_augmentation_log(self, originals, generated):
        """Save augmentation process log"""
        log_data = {
            "original_samples": originals,
            "generated_samples": generated,
            "total_samples": originals + generated,
            "augmentations_per_sample": self.augmentations_per_sample,
            "input_folder": str(self.input_folder),
            "output_folder": str(self.output_folder),
            "classes": self.classes
        }
        
        log_file = self.output_folder / "augmentation_log.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📝 Augmentation log saved: {log_file}")

def main():
    """Main function"""
    print("🎸 Dataset Augmentation Tool")
    print("=" * 50)
    
    augmenter = DatasetAugmenter(
        input_folder="training_data_classified",
        output_folder="training_data_augmented"
    )
    
    print("\nSelect augmentation mode:")
    print("1. Balanced dataset (2000 samples per class)")
    print("2. Balanced dataset (5000 samples per class)")
    print("3. Fixed augmentations (25 per original)")
    print("4. Custom target")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        augmenter.run_augmentation(balanced=True, target_per_class=2000)
    elif choice == '2':
        augmenter.run_augmentation(balanced=True, target_per_class=5000)
    elif choice == '3':
        augmenter.run_augmentation(balanced=False)
    elif choice == '4':
        target = int(input("Enter target samples per class: "))
        augmenter.run_augmentation(balanced=True, target_per_class=target)
    else:
        print("Invalid choice")
        return
    
    print("\n🎯 Next steps:")
    print("1. Update train_model.py to use 'training_data_augmented' folder")
    print("2. Run 'python train_model.py' to train with the expanded dataset")

if __name__ == "__main__":
    main() 