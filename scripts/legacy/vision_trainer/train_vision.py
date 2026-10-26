import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os

class VisionCNN(nn.Module):
    def __init__(self, num_classes=10, embedding_dim=512):
        super(VisionCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 768, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(768, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        features = self.embedding(x)
        logits = self.classifier(features)
        return logits, features

class SyntheticShapeDataset(Dataset):
    def __init__(self, num_classes=10, samples_per_class=10, image_size=64, transform=None):
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform
        self.images = []
        self.labels = []
        
        np.random.seed(42)
        
        for cls in range(num_classes):
            for _ in range(samples_per_class):
                image = self.generate_class_image(cls)
                self.images.append(image)
                self.labels.append(cls)
        
        self.images = np.array(self.images, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        indices = np.random.permutation(len(self.labels))
        self.images = self.images[indices]
        self.labels = self.labels[indices]
    
    def generate_class_image(self, class_idx):
        image = np.ones((3, self.image_size, self.image_size), dtype=np.float32) * -0.8
        
        cx = self.image_size * 0.5 + (np.random.rand() - 0.5) * self.image_size * 0.1
        cy = self.image_size * 0.5 + (np.random.rand() - 0.5) * self.image_size * 0.1
        size = self.image_size * 0.3 * (1 + (np.random.rand() - 0.5) * 0.2)
        
        shape_type = class_idx % 10
        
        for y in range(self.image_size):
            for x in range(self.image_size):
                dx = x - cx
                dy = y - cy
                dist = np.sqrt(dx * dx + dy * dy)
                draw = False
                
                if shape_type == 0:
                    draw = dist < size
                    r, g, b = 0.9, 0.2, 0.2
                elif shape_type == 1:
                    draw = np.abs(dx) < size/2 and np.abs(dy) < size/2
                    r, g, b = 0.2, 0.8, 0.2
                elif shape_type == 2:
                    draw = (cy - y) >= 0 and (cy - y) <= size and np.abs(dx) <= (1 - (cy - y)/size) * size/2
                    r, g, b = 0.2, 0.3, 0.9
                elif shape_type == 3:
                    thickness = size * 0.25
                    draw = ((np.abs(dx) < thickness/2 and np.abs(dy) < size/2) or 
                            (np.abs(dy) < thickness/2 and np.abs(dx) < size/2))
                    r, g, b = 0.9, 0.9, 0.2
                elif shape_type == 4:
                    draw = np.abs(dx)/(size/2) + np.abs(dy)/(size/2) <= 1
                    r, g, b = 0.9, 0.5, 0.1
                elif shape_type == 5:
                    draw = dist < size and dist > size * 0.55
                    r, g, b = 0.8, 0.3, 0.8
                elif shape_type == 6:
                    stripe_h = max(2, self.image_size // 10)
                    offset = (self.image_size - size) / 2
                    if x >= offset and x <= self.image_size - offset and y >= offset and y <= self.image_size - offset:
                        draw = ((int)((y - offset) // stripe_h)) % 2 == 0
                    r, g, b = 0.5, 0.5, 0.5
                elif shape_type == 7:
                    stripe_w = max(2, self.image_size // 10)
                    offset = (self.image_size - size) / 2
                    if x >= offset and x <= self.image_size - offset and y >= offset and y <= self.image_size - offset:
                        draw = ((int)((x - offset) // stripe_w)) % 2 == 0
                    r, g, b = 0.9, 0.6, 0.3
                elif shape_type == 8:
                    cell = max(3, self.image_size // 8)
                    offset = (self.image_size - size) / 2
                    if x >= offset and x <= self.image_size - offset and y >= offset and y <= self.image_size - offset:
                        cell_x = (int)((x - offset) // cell)
                        cell_y = (int)((y - offset) // cell)
                        draw = (cell_x + cell_y) % 2 == 0
                    r, g, b = 0.7, 0.7, 0.7
                else:
                    dot_spacing = max(4, self.image_size // 12)
                    offset = (self.image_size - size) / 2
                    if x >= offset and x <= self.image_size - offset and y >= offset and y <= self.image_size - offset:
                        dx_dot = (int)((x - offset) % dot_spacing) - dot_spacing // 2
                        dy_dot = (int)((y - offset) % dot_spacing) - dot_spacing // 2
                        draw = dx_dot * dx_dot + dy_dot * dy_dot < dot_spacing * dot_spacing * 0.12
                    r, g, b = 0.4, 0.8, 0.6
                
                if draw:
                    image[0, y, x] = r
                    image[1, y, x] = g
                    image[2, y, x] = b
        
        noise = (np.random.rand(3, self.image_size, self.image_size) - 0.5) * 2 * 0.08
        image = np.clip(image + noise, -1.0, 1.0)
        
        return image
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return torch.from_numpy(image), torch.tensor(label)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("=" * 70)
    print("VISION MODEL TRAINING WITH PyTorch (AMD 680M GPU)")
    print("=" * 70)
    
    device = torch.device("cpu")
    device_name = "CPU"
    device_memory = 0
    
    try:
        import torch_directml
        dml_device = torch_directml.device()
        device = dml_device
        device_name = "DirectML (AMD Radeon 680M)"
        print(f"✓ DirectML device detected: {device}")
    except ImportError:
        print("✗ torch-directml not installed, trying CUDA...")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print(f"✓ CUDA device detected: {device_name}")
        else:
            print("✗ Neither DirectML nor CUDA available, using CPU")
    
    print(f"Device: {device}")
    print(f"Device Name: {device_name}")
    if device_memory > 0:
        print(f"Device Memory: {device_memory:.1f} MB")
    
    num_classes = 10
    image_size = 64
    embedding_dim = 512
    batch_size = 8
    epochs = 15
    learning_rate = 0.005
    
    print(f"\nTraining Config:")
    print(f"  Classes: {num_classes}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    
    model = VisionCNN(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    total_params = count_parameters(model)
    print(f"\nModel Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    
    dataset = SyntheticShapeDataset(
        num_classes=num_classes,
        samples_per_class=20,
        image_size=image_size
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Training samples: {len(dataset)}")
    print("=" * 70)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            logits, _ = model(images)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataset)
        accuracy = correct / total * 100
        
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}% | Time: {elapsed:.1f}s")
        
        scheduler.step(avg_loss)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Initial accuracy: {accuracies[0]:.1f}%")
    print(f"Final accuracy: {accuracies[-1]:.1f}%")
    if losses[0] > 0:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"Loss improvement: {improvement:.2f}%")
    
    save_dir = "G:/青小团多模态语言大模型 Qingxiaotuan LLM v1.0/models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vision_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()