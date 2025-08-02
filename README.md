# Transfer Learning for Image Classification of Five Flower Classes

This project demonstrates the use of transfer learning to build an image classification model for five distinct classes of flowers. The model fine-tunes a pre-trained **ResNet50** model using the PyTorch framework.

## Dataset

The dataset used for this project consists of five classes of flowers. It was generously provided by Vizuara AI Labs.

- **Dataset Source:** [Vizuara AI Labs Dataset (Google Drive)](https://drive.google.com/drive/folders/1BiqW9HEl3Ld-ik1LefD0YQFToRdrmQd1)

The dataset is expected to be a zipped file named `5flowers-dataset.zip` which contains `train` and `val` subdirectories for the respective flower classes.

## Model Architecture

The core of this project is a transfer learning approach using the following architecture:

- **Base Model:** A pre-trained `ResNet50` model, originally trained on the ImageNet dataset, serves as the base for feature extraction. The weights of the base model's layers are frozen to leverage the powerful features it has already learned.

- **Custom Classifier Head:** A new, custom classification head is added on top of the frozen `ResNet50` base. This head is designed to handle the specific task of classifying the five flower classes. It consists of:
    - A dense hidden layer with `256` units and a `ReLU` activation function.
    - A final output layer with `5` units, corresponding to the five distinct flower classes.

## Dependencies

The following Python libraries are required to run the Jupyter Notebook:

- `torch`
- `torchvision`
- `torchinfo`
- `numpy`
- `pandas`
- `matplotlib`
-  `PIL`

You can install these dependencies using `pip`:

```bash
!pip install -q torch torchvision torchinfo numpy pandas matplotlib 