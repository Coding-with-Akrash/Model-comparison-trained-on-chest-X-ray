# Transfer Learning for Medical Image Classification

This project demonstrates the application of transfer learning techniques for classifying medical images using both PyTorch and TensorFlow frameworks. It focuses on pneumonia detection in chest X-rays but includes support for multiple medical imaging datasets.

## Features

- **Multi-Framework Support**: Implements transfer learning using both PyTorch and TensorFlow
- **Multiple Pre-trained Models**: Includes popular architectures like ResNet, VGG, EfficientNet, MobileNet, ViT, and DenseNet
- **Comprehensive Evaluation**: Provides detailed model comparison with metrics, confusion matrices, and ROC curves
- **Datasets**: Supports medical imaging dataset chest X-rays
- **Automated Training**: Scripts for batch training across different models and datasets
- **Visualization**: Generates plots for training history, confusion matrices, and performance comparisons

## Datasets

The project supports the following medical imaging datasets:

- **Chest X-Ray**: Pneumonia detection (NORMAL vs PNEUMONIA)

## Models

### PyTorch Models
- ResNet18
- DenseNet121
- EfficientNet-B0
- MobileNetV2
- VGG16
- Vision Transformer (ViT-B/16)

### TensorFlow/Keras Models
- VGG16
- EfficientNetB0
- Custom ResNet
- Custom EfficientNet
- Custom CNN

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Coding-with-Akrash/Model-comparison-trained-on-chest-X-ray.git
cd transfer-learning-medical
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install tensorflow keras
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python
```

## Usage

### Training Models

#### PyTorch Training
```bash
python pytorch_training.py
```

#### TensorFlow Training
```bash
python tensorflow_training.py
```

### Model Comparison and Evaluation

```bash
python model_comparison.py
```

This will:
- Load trained models from the `trained_models/` directory
- Evaluate models on test data
- Generate performance metrics (accuracy, precision, recall, F1-score)
- Create confusion matrices for each model
- Generate ROC curves for binary classification
- Produce an HTML report with embedded visualizations

### Individual Model Evaluation

```bash
python comp.py
```

Similar to model_comparison.py but with simplified output.

## Project Structure

```
├── dataset/
│   └── TRAIN/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── models/
│   ├── resnet18_model.py
│   ├── vgg16_model.py
│   ├── ...
│   └── keras_*.py
├── trained_models/
│   ├── resnet18_model.pth
│   ├── vgg16_model.pth
│   └── ...
├── pytorch_training.py
├── tensorflow_training.py
├── model_comparison.py
├── comp.py
├── LICENSE
└── README.md
```

## Results

After running the evaluation scripts, you'll find:

- `model_comparison_results.csv`: Tabular results of model performance
- `model_comparison_report.html`: Interactive HTML report with visualizations
- Individual confusion matrix images: `{model_name}_confusion_matrix.png`
- ROC curve comparison: `roc_curves_comparison.png`
- Training history plots: `{model_name}_{dataset}_history.png`

## Key Findings

The project compares different transfer learning approaches across medical imaging tasks, demonstrating:

- Performance variations between different pre-trained architectures
- Effectiveness of transfer learning for medical image classification
- Comparative analysis of PyTorch vs TensorFlow implementations
- Visualization of model predictions and evaluation metrics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Akrash Noor
Hifzun Nisa

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer


This project is for educational and research purposes. The models and results should not be used for actual medical diagnosis without proper validation and clinical evaluation.

