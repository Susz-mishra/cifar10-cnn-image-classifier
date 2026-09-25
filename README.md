# 🖼️ CIFAR-10 CNN Image Classifier

A Convolutional Neural Network (CNN) built using **PyTorch** to classify images from the **CIFAR-10 dataset** into 10 different classes.

## 📌 Project Overview

This project implements a CNN from scratch using PyTorch for image classification on the CIFAR-10 dataset.

The complete workflow includes:

* Loading the CIFAR-10 dataset
* Image preprocessing and normalization
* Creating PyTorch DataLoaders
* Building a custom CNN architecture
* Training the model using backpropagation
* Evaluating the model on the test dataset
* Saving the trained model

## 🎯 Objective

The main objective of this project is to understand and implement the fundamental concepts of **Convolutional Neural Networks** for multi-class image classification using PyTorch.

## 📊 Dataset

The project uses the **CIFAR-10 dataset**, which contains 60,000 RGB images of size **32 × 32 pixels** belonging to 10 classes.

The classes are:

* ✈️ Airplane
* 🚗 Automobile
* 🐦 Bird
* 🐱 Cat
* 🦌 Deer
* 🐶 Dog
* 🐸 Frog
* 🐴 Horse
* 🚢 Ship
* 🚚 Truck

The dataset is divided into:

* **50,000 training images**
* **10,000 test images**

## 🔄 Project Workflow

```text
CIFAR-10 Dataset
       ↓
Image Preprocessing
       ↓
Normalization
       ↓
PyTorch DataLoader
       ↓
CNN Architecture
       ↓
Forward Propagation
       ↓
Cross Entropy Loss
       ↓
Backpropagation
       ↓
Adam Optimizer
       ↓
Model Evaluation
       ↓
Test Accuracy
       ↓
Save Trained Model
```

## 🧹 Data Preprocessing

The images are converted to PyTorch tensors using `ToTensor()` and normalized using:

```python
transforms.Normalize(
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)
)
```

The data is loaded using PyTorch `DataLoader` with:

```text
Batch Size = 64
Training Data = Shuffled
Test Data = Not Shuffled
```

## 🧠 CNN Architecture

The CNN consists of three convolutional blocks followed by fully connected layers.

```text
Input
32 × 32 × 3
      ↓
Conv2D: 3 → 32
      ↓
ReLU
      ↓
MaxPool 2×2
      ↓
Conv2D: 32 → 64
      ↓
ReLU
      ↓
MaxPool 2×2
      ↓
Conv2D: 64 → 128
      ↓
ReLU
      ↓
MaxPool 2×2
      ↓
Flatten
      ↓
Linear: 2048 → 256
      ↓
ReLU
      ↓
Linear: 256 → 10
      ↓
Class Prediction
```

### Model Configuration

| Component       | Configuration         |
| --------------- | --------------------- |
| Input           | 32 × 32 × 3 RGB image |
| Conv Layer 1    | 3 → 32 filters        |
| Conv Layer 2    | 32 → 64 filters       |
| Conv Layer 3    | 64 → 128 filters      |
| Activation      | ReLU                  |
| Pooling         | MaxPool2d(2,2)        |
| Fully Connected | 2048 → 256            |
| Output Layer    | 256 → 10              |
| Output Classes  | 10                    |

## ⚙️ Training Configuration

The model was trained using:

| Parameter     | Value            |
| ------------- | ---------------- |
| Epochs        | 10               |
| Batch Size    | 64               |
| Loss Function | CrossEntropyLoss |
| Optimizer     | Adam             |
| Dataset       | CIFAR-10         |
| Framework     | PyTorch          |

## 📈 Training Results

The training and validation loss recorded during training were:

| Epoch | Training Loss | Validation Loss |
| ----: | ------------: | --------------: |
|     1 |        1.3686 |          1.0687 |
|     2 |        0.9411 |          0.8701 |
|     3 |        0.7590 |          0.7902 |
|     4 |        0.6343 |          0.7378 |
|     5 |        0.5292 |          0.7391 |
|     6 |        0.4410 |          0.7468 |
|     7 |        0.3582 |          0.7773 |
|     8 |        0.2875 |          0.8332 |
|     9 |        0.2283 |          0.9345 |
|    10 |        0.1770 |          1.0704 |

## 🎯 Model Performance

The trained CNN achieved:

> **Test Accuracy: 75.32%**

The model was evaluated on the CIFAR-10 test dataset after completing 10 training epochs.

## 💾 Model Saving

The trained model parameters are saved using PyTorch:

```python
torch.save(model.state_dict(), "cnn_cifar10.pth")
```

Saved model:

```text
cnn_cifar10.pth
```

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* CIFAR-10
* Jupyter Notebook

## 📁 Project Structure

```text
cifar10-cnn-image-classifier/
│
├── dataCNN/
│   └── CIFAR-10 dataset
│
├── CNN_CIFAR10.ipynb
│
├── cnn_cifar10.pth
│
├── README.md
│
└── requirements.txt
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Susz-mishra/cifar10-cnn-image-classifier.git
```

### 2. Navigate to the project directory

```bash
cd cifar10-cnn-image-classifier
```

### 3. Install dependencies

```bash
pip install torch torchvision
```

Or install all dependencies using:

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

Open:

```text
CNN_CIFAR10.ipynb
```

using Jupyter Notebook or JupyterLab and execute the cells.

## 📚 Key Learning Outcomes

Through this project, I practiced:

* Building CNNs using PyTorch
* Working with image datasets
* Image normalization
* PyTorch tensors and DataLoaders
* Convolutional layers
* ReLU activation
* Max pooling
* Fully connected layers
* Forward propagation
* Backpropagation
* Loss functions
* Adam optimization
* Model evaluation
* Saving trained PyTorch models

## 🚀 Future Improvements

Possible improvements include:

* Data augmentation
* Dropout regularization
* Batch normalization
* Learning-rate scheduling
* Hyperparameter tuning
* Training for more epochs with appropriate regularization
* Visualization of training/validation curves
* Per-class performance analysis
* Experimenting with deeper CNN architectures

## 👨‍💻 Author

**Anuj Kumar Mishra**

Computer Science Student | AI/ML & Deep Learning Enthusiast
