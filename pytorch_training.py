import os
import sys
import subprocess

# Define datasets that work with ImageFolder
datasets = {
    'Brain Tumor': 'dataset/Brain tumor',
    'Blood Cells': 'dataset/Blood Cells'
}

# PyTorch models
pytorch_models = [
    'resnet18_model.py',
    'densenet121_model.py',
    'efficientnet_b0_model.py',
    'mobilenet_v2_model.py',
    'vgg16_model.py',
    'vit_b_16_model.py'
]

for dataset_name, dataset_path in datasets.items():
    print(f"Processing {dataset_name} with PyTorch models")
    for model_file in pytorch_models:
        print(f"Training {model_file} on {dataset_name}")
        try:
            subprocess.run([sys.executable, os.path.join('models', model_file), dataset_path, '1'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error training {model_file} on {dataset_name}: {e}")

print("PyTorch training completed.")