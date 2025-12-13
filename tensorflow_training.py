import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from models.keras_resnet_model import create_keras_resnet_model
from models.keras_efficientnet_model import create_keras_efficientnet_model
from models.keras_cnn_model import create_keras_cnn_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import cv2

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define datasets and their paths
datasets = {
    'Brain Tumor': 'dataset/Brain tumor',
    'Skin Cancer': 'dataset/Skin cancer',
    'Blood Cells': 'dataset/Blood Cells',
    'Chest X-Ray': 'dataset/cheast X-rays'
}

# Define models
models_dict = {
    'VGG16': VGG16,
    'EfficientNetB0': EfficientNetB0,
    'KerasResNet': create_keras_resnet_model,
    'KerasEfficientNet': create_keras_efficientnet_model,
    'KerasCNN': create_keras_cnn_model
}

# Image size for all models
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1  # Reduce for demo, increase as needed

# Function to load and preprocess data
def load_data(dataset_path, dataset_name):
    if dataset_name == 'Brain Tumor':
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_gen = datagen.flow_from_directory(
            dataset_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training',
            classes=['no', 'yes']
        )
        val_gen = datagen.flow_from_directory(
            dataset_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation',
            classes=['no', 'yes']
        )
        num_classes = 2
    elif dataset_name == 'Blood Cells':
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_gen = datagen.flow_from_directory(
            dataset_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        val_gen = datagen.flow_from_directory(
            dataset_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
        num_classes = len(train_gen.class_indices)
    elif dataset_name == 'Chest X-Ray':
        # Use CSV for labels
        csv_path = os.path.join(dataset_path, 'labels_train.csv')
        img_dir = os.path.join(dataset_path, 'train_images')
        df = pd.read_csv(csv_path)
        df['class_id'] = df['class_id'].astype(str)  # Convert to string
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_gen = datagen.flow_from_dataframe(
            dataframe=df,
            directory=img_dir,
            x_col='file_name',
            y_col='class_id',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        val_gen = datagen.flow_from_dataframe(
            dataframe=df,
            directory=img_dir,
            x_col='file_name',
            y_col='class_id',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
        num_classes = len(train_gen.class_indices)
    elif dataset_name == 'Skin Cancer':
        # Use CSV for labels
        csv_path = os.path.join(dataset_path, 'HAM10000_metadata.csv')
        df = pd.read_csv(csv_path)
        df['dx'] = df['dx'].astype(str)
        # Add full path
        df['image_path'] = df['image_id'].apply(lambda x: os.path.join(dataset_path, 'HAM10000_images_part_1', x + '.jpg') if os.path.exists(os.path.join(dataset_path, 'HAM10000_images_part_1', x + '.jpg')) else os.path.join(dataset_path, 'HAM10000_images_part_2', x + '.jpg'))
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        datagen = ImageDataGenerator(rescale=1./255)
        train_gen = datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='image_path',
            y_col='dx',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        val_gen = datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col='image_path',
            y_col='dx',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        num_classes = len(train_gen.class_indices)
    return train_gen, val_gen, num_classes

# Function to build model
def build_model(base_model_class, num_classes, input_shape=IMG_SIZE + (3,)):
    if base_model_class in [ResNet50, VGG16, EfficientNetB0]:
        base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False  # Freeze base model
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        if num_classes == 2:
            output = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            output = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        model = Model(inputs=base_model.input, outputs=output)
    else:
        # Custom keras models
        model = base_model_class(input_shape, num_classes)
        if num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    return model

# Function to visualize dataset
def visualize_dataset(dataset_path, dataset_name, num_samples=5):
    if dataset_name == 'Brain Tumor':
        classes = ['no', 'yes']
    elif dataset_name == 'Blood Cells':
        classes = ['LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
    elif dataset_name == 'Chest X-Ray':
        classes = ['NORMAL', 'PNEUMONIA']  # Assuming
    elif dataset_name == 'Skin Cancer':
        # For skin cancer, classes from metadata
        metadata = pd.read_csv(os.path.join(dataset_path, 'HAM10000_metadata.csv'))
        classes = metadata['dx'].unique()
    else:
        classes = os.listdir(dataset_path)

    fig, axes = plt.subplots(len(classes), num_samples, figsize=(num_samples*2, len(classes)*2))
    for i, cls in enumerate(classes):
        if dataset_name == 'Brain Tumor':
            cls_path = os.path.join(dataset_path, cls)
        elif dataset_name == 'Blood Cells':
            cls_path = os.path.join(dataset_path, cls)
        elif dataset_name == 'Chest X-Ray':
            # Assuming train_images has subfolders
            cls_path = os.path.join(dataset_path, 'train_images', cls)
        elif dataset_name == 'Skin Cancer':
            # This is complex; for simplicity, skip or use part_1
            continue
        if os.path.exists(cls_path):
            images = os.listdir(cls_path)[:num_samples]
            for j, img_name in enumerate(images):
                img_path = os.path.join(cls_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                    if j == 0:
                        axes[i, j].set_title(cls)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_samples.png')
    plt.show()

# Function to plot training history
def plot_history(history, model_name, dataset_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='val')
    ax1.set_title(f'{model_name} on {dataset_name} - Accuracy')
    ax1.legend()
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='val')
    ax2.set_title(f'{model_name} on {dataset_name} - Loss')
    ax2.legend()
    plt.savefig(f'{model_name}_{dataset_name}_history.png')
    plt.show()

# Main training loop
results = {}
for dataset_name, dataset_path in datasets.items():
    print(f"Processing {dataset_name}")
    # Visualize dataset
    visualize_dataset(dataset_path, dataset_name)

    # Load data
    train_gen, val_gen, num_classes = load_data(dataset_path, dataset_name)

    for model_name, model_class in models_dict.items():
        print(f"Training {model_name} on {dataset_name}")
        model = build_model(model_class, num_classes)

        # # Plot model architecture before training
        # plot_model(model, to_file=f'{model_name}_{dataset_name}_before.png', show_shapes=True)

        # Train
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS
        )

        # Save model
        os.makedirs('trained_models', exist_ok=True)
        if model_class in [ResNet50, VGG16, EfficientNetB0]:
            try:
                model.save(os.path.join('trained_models', f'{model_name}_{dataset_name}.h5'))
            except Exception as e:
                print(f"Error saving model {model_name}_{dataset_name}: {e}")
        else:
            # For custom models, save weights
            try:
                model.save_weights(os.path.join('trained_models', f'{model_name}_{dataset_name}_weights.h5'))
            except Exception as e:
                print(f"Error saving weights for {model_name}_{dataset_name}: {e}")

        # Plot history
        plot_history(history, model_name, dataset_name)

        # Evaluate
        val_loss, val_acc = model.evaluate(val_gen)
        results[(model_name, dataset_name)] = {'accuracy': val_acc, 'loss': val_loss}

        # Get predictions for confusion matrix and ROC
        val_gen.reset()
        y_pred_prob = model.predict(val_gen)
        y_true = val_gen.classes

        if num_classes == 2:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(val_gen.class_indices.keys()), yticklabels=list(val_gen.class_indices.keys()))
        plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'{model_name}_{dataset_name}_confusion_matrix.png')
        plt.close()

        # ROC curve for binary classification
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob.flatten())
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name} on {dataset_name}')
            plt.legend(loc="lower right")
            plt.savefig(f'{model_name}_{dataset_name}_roc_curve.png')
            plt.close()

# Comparison
results_df = pd.DataFrame(results).T
print(results_df)

# Plot comparison
results_df.reset_index(inplace=True)
results_df.columns = ['Model', 'Dataset', 'Accuracy', 'Loss']

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=results_df, x='Dataset', y='Accuracy', hue='Model', ax=ax)
plt.title('Model Accuracy Comparison')
plt.savefig('accuracy_comparison.png')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=results_df, x='Dataset', y='Loss', hue='Model', ax=ax)
plt.title('Model Loss Comparison')
plt.savefig('loss_comparison.png')
plt.show()

# Save results to CSV
results_df.to_csv('model_comparison.csv', index=False)