# MNIST Digit Recognition

This project trains neural networks to recognize handwritten digits from the MNIST dataset. It is written as a Jupyter notebook using PyTorch and demonstrates three important machine learning ideas:

- how a compact baseline model behaves on MNIST
- how a very large model can overfit training data
- how regularization can improve performance on unseen test data

The main notebook is [`mnist_digit_recognition.ipynb`](mnist_digit_recognition.ipynb).

## Project Overview

The project uses the MNIST dataset, which contains 70,000 grayscale images of handwritten digits from 0 to 9. Each image is 28 x 28 pixels.

The notebook follows four stages:

1. Data pipeline: load MNIST, normalize images, and create train, validation, and test loaders.
2. Baseline model: train a compact model with light dropout regularization.
3. Overfitting experiment: train a much larger fully connected model without regularization.
4. Regularized experiment: train the same large architecture again with augmentation, lower learning rate, and L2 weight decay.

The goal is to compare a healthy baseline, an intentionally overfit model, and a regularized version of the same large model.

## Dataset Split

| Split | Samples | Purpose |
|---|---:|---|
| Training | 50,000 | Used to update model weights |
| Validation | 10,000 | Used to monitor generalization during training |
| Test | 10,000 | Used only for final evaluation |

Batch size: **256**.

## Model Architectures

### BaselineNet

`BaselineNet` is the normal reference model. It is compact and includes dropout to reduce overfitting.

```text
Input image: 1 x 28 x 28
Flatten: 784 features
Linear: 784 -> 128, ReLU
Dropout: p = 0.2
Linear: 128 -> 10
Output: logits for digits 0-9
```

Total trainable parameters: **101,770**.

### OverfitNet

`OverfitNet` is a much larger fully connected model. The same architecture is used in Stage 3 and Stage 4; only the training procedure changes.

```text
Input image: 1 x 28 x 28
Flatten: 784 features
Linear: 784 -> 2048, ReLU
Linear: 2048 -> 2048, ReLU
Linear: 2048 -> 2048, ReLU
Linear: 2048 -> 2048, ReLU
Linear: 2048 -> 1024, ReLU
Linear: 1024 -> 10
Output: logits for digits 0-9
```

Total trainable parameters: **16,305,162**.

## Training Setup

| Stage | Model | Epochs | Learning rate | Weight decay | Augmentation |
|---|---|---:|---:|---:|---|
| Stage 2 | BaselineNet | 30 | 0.0005 | 0 | None |
| Stage 3 | OverfitNet | 50 | 0.001 | 0 | None |
| Stage 4 | OverfitNet | 50 | 0.0005 | 0.0001 | Random rotation and translation |

All stages use `CrossEntropyLoss` and the Adam optimizer.

## Results

| Metric | Stage 2: BaselineNet | Stage 3: OverfitNet | Stage 4: Regularized OverfitNet |
|---|---:|---:|---:|
| Final train loss | 0.0280 | 0.0072 | 0.0452 |
| Final validation loss | 0.0781 | 0.1928 | 0.0359 |
| Final train accuracy | 99.14% | 99.86% | 98.51% |
| Final validation accuracy | 97.65% | 98.05% | 99.02% |
| Train-validation accuracy gap | 1.49 percentage points | 1.81 percentage points | -0.51 percentage points |
| Test accuracy | 97.90% | 98.32% | 99.04% |

The overfit model achieves the highest training accuracy, but its validation loss is much higher. The regularized model gives the best held-out test accuracy, improving from **98.32%** to **99.04%** over the unregularized overfit model.

## Generated Outputs

The notebook generates these result images:

- [`figures/sample.png`](figures/sample.png): sample MNIST training images.
- [`figures/stage2.png`](figures/stage2.png): Stage 2 baseline training and validation curves.
- [`figures/stage3.png`](figures/stage3.png): Stage 3 overfitting training and validation curves.
- [`figures/stage4.png`](figures/stage4.png): Stage 4 regularized training and validation curves.
- [`figures/loss_curves.png`](figures/loss_curves.png): loss comparison for the overfit and regularized models.
- [`figures/fig5_loss_comparison.png`](figures/fig5_loss_comparison.png): overlaid loss curves for baseline, overfit, and regularized models.
- [`figures/fig6_accuracy_comparison.png`](figures/fig6_accuracy_comparison.png): overlaid accuracy curves for baseline, overfit, and regularized models.
- [`figures/per_class_accuracy.png`](figures/per_class_accuracy.png): accuracy for each digit class.
- [`figures/confusion_matrix.png`](figures/confusion_matrix.png): confusion matrix for the regularized model's test predictions.

Preview:

![Sample MNIST digits](figures/sample.png)

![Baseline training curves](figures/stage2.png)

![Loss comparison](figures/fig5_loss_comparison.png)

![Accuracy comparison](figures/fig6_accuracy_comparison.png)

![Confusion matrix](figures/confusion_matrix.png)

The university report is available at [`report/MNIST.docx`](report/MNIST.docx).

## Project Structure

```text
mnstproj/
|-- .gitignore                           # Ignores local data and virtual environment files
|-- data/
|   `-- MNIST/
|       `-- raw/                         # Downloaded MNIST files
|-- figures/
|   |-- confusion_matrix.png             # Confusion matrix for test predictions
|   |-- fig5_loss_comparison.png         # Baseline vs overfit vs regularized loss curves
|   |-- fig6_accuracy_comparison.png     # Baseline vs overfit vs regularized accuracy curves
|   |-- loss_curves.png                  # Stage 3 vs Stage 4 loss curves
|   |-- per_class_accuracy.png           # Accuracy by digit class
|   |-- sample.png                       # Example MNIST images from a training batch
|   |-- stage2.png                       # Stage 2 baseline chart
|   |-- stage3.png                       # Stage 3 overfitting chart
|   `-- stage4.png                       # Stage 4 regularized training chart
|-- mnist_env/                           # Local Python virtual environment
|-- report/
|   `-- MNIST.docx                       # University submission report
|-- mnist_digit_recognition.ipynb        # Main project notebook
|-- README.md                            # Project documentation
`-- requirements.txt                     # Python package dependencies
```

## Requirements

The project uses:

- Python 3.11
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook or JupyterLab

The folder already contains a local virtual environment named `mnist_env`.
Dependencies are listed in [`requirements.txt`](requirements.txt).

## How to Run

### Option 1: Use the existing virtual environment

In PowerShell:

```powershell
.\mnist_env\Scripts\Activate.ps1
jupyter notebook mnist_digit_recognition.ipynb
```

Then run the notebook cells from top to bottom.

### Option 2: Create a fresh environment

```powershell
python -m venv mnist_env
.\mnist_env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook mnist_digit_recognition.ipynb
```

## Reproducibility Notes

- The notebook uses a fixed random seed.
- The train/validation split is fixed so all stages are compared fairly.
- `BaselineNet` establishes normal learning behavior with a compact network and dropout.
- `OverfitNet` is used unchanged in Stage 3 and Stage 4.
- Stage 4 changes only the training procedure, not the large model architecture.

## Key Learning Outcome

The main lesson is that high training accuracy alone does not prove that a model is best. The baseline model gives a healthy reference, the overfit model shows the risk of excessive capacity, and the regularized model shows how data augmentation, weight decay, and a lower learning rate can improve generalization.
