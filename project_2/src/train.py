import os
import argparse

import torch
import onnx
import torch.nn as nn
import torch.optim as optim
from data_preprocessing import get_data_loaders, analyze_dataset
from model import get_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from tqdm import tqdm


def export_to_onnx(model, model_path, device):
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy_input, model_path, verbose=True, input_names=['input'], output_names=['output'])
    print(f"Model exported to {model_path}")

def train_model(data_dir, model_name, num_epochs=50, batch_size=32, learning_rate=0.001, freeze_base=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    analyze_dataset(data_dir)

    train_loader, val_loader, class_to_idx = get_data_loaders(data_dir, batch_size)
    num_classes = len(class_to_idx)
    
    model = get_model(num_classes, model_name).to(device)
    
    if freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'models/{model_name}')
    os.makedirs(models_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(models_dir, f'best_model_{model_name}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, f'training_plots_{model_name}.png'))
    plt.close()

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(models_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()

    # Save the final model
    final_model_path = os.path.join(models_dir, f'final_model_{model_name}.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    onnx_model_path = os.path.join(models_dir, f'final_model_{model_name}.onnx')
    export_to_onnx(model, onnx_model_path, device)

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model {onnx_model_path} is valid")

    return class_to_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train car angle classifier")
    parser.add_argument("--data_dir", type=str, default="data/", help="Path to the dataset")
    parser.add_argument("--model", type=str, default="resnet50", 
                        choices=['resnet50', 'resnet18', 'efficientnet_b0', 'efficientnet_v2_s', 'vit_b_16', 'regnet_y_32gf'], 
                        help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--freeze_base", action="store_true", help="Freeze the base model and only train the prediction head")

    args = parser.parse_args()

    train_model(args.data_dir, args.model, args.epochs, args.batch_size, args.learning_rate, args.freeze_base)
