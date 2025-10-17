# Machine Learning Tools

This repository contains small machine learning demos, scripts, models, and notebooks used for learning and experimentation.It also contains the first part of the assignment in (docs/part 1.pdf) 

## Project structure (selected)

- `app.py` — Streamlit web application for MNIST digit classification
- `src/` — source code and models
- `assets/` — images, videos, and HTML outputs
- `src/notebooks/` — Jupyter notebooks for each task
- `src/webapp/` — Web application source code
- `src/models/` — Trained machine learning models

## Models Overview

### MNIST CNN Model (`mnist_cnn_model.h5`)
acess it here https://machine-learning-tools-cqetrqnappj9uqsxrlabahf.streamlit.app/
- **Type**: Convolutional Neural Network (CNN)
- **Purpose**: Handwritten digit recognition (0-9)
- **Architecture**:
  - Input: 28×28×1 grayscale images
  - 2 Convolutional layers with MaxPooling
  - Fully connected layers with Dropout
  - Output: 10 classes with softmax activation
- **Performance**: >98% accuracy on MNIST test set
- **Framework**: TensorFlow/Keras
- **Usage**: Real-time digit classification via web interface

### Iris Decision Tree Model
- **Type**: Decision Tree Classifier
- **Purpose**: Iris flower species classification
- **Features**: Sepal length, sepal width, petal length, petal width
- **Classes**: Setosa, Versicolor, Virginica
- **Performance**: 100% accuracy on test set
- **Framework**: Scikit-learn
- **Usage**: Classical machine learning demonstration

### spaCy NLP Model
- **Type**: Pre-trained language model
- **Purpose**: Named Entity Recognition (NER) and text processing
- **Capabilities**:
  - Entity extraction (products, brands, organizations)
  - Rule-based sentiment analysis
  - Text preprocessing and analysis
- **Model**: `en_core_web_sm` (English small model)
- **Framework**: spaCy
- **Usage**: Amazon reviews analysis and NLP tasks

## Model Training Details

### MNIST CNN Training
- **Dataset**: 60,000 training images, 10,000 test images
- **Preprocessing**: Normalization (0-1), reshaping
- **Training**: 15 epochs with Adam optimizer
- **Validation**: 10% split from training data
- **Metrics**: Accuracy, precision, recall, confusion matrix

### Iris Classifier Training
- **Dataset**: 150 samples (50 per class)
- **Preprocessing**: Standard scaling, label encoding
- **Algorithm**: Decision Tree with max_depth=3
- **Evaluation**: Cross-validation, classification report

### NLP Processing
- **Data**: Amazon product reviews
- **Techniques**: Rule-based sentiment analysis, entity recognition
- **Visualization**: NER displays, sentiment distribution charts

## Assets

Images

- [assets/images/mnist_training_history.png](assets/images/mnist_training_history.png) - Training loss and accuracy curves
- [assets/images/mnist_sample_images.png](assets/images/mnist_sample_images.png) - Sample digits from MNIST dataset
- [assets/images/mnist_predictions.png](assets/images/mnist_predictions.png) - Model predictions on test samples
- [assets/images/mnist_confusion_matrix.png](assets/images/mnist_confusion_matrix.png) - Classification performance matrix
- [assets/images/iris_distributions.png](assets/images/iris_distributions.png) - Feature distributions by species
- [assets/images/iris_correlation.png](assets/images/iris_correlation.png) - Feature correlation heatmap
- [assets/images/iris_confusion_matrix.png](assets/images/iris_confusion_matrix.png) - Iris classifier performance
- [assets/images/amazon_reviews_analysis.png](assets/images/amazon_reviews_analysis.png) - NLP analysis results

Videos

- [assets/videos/presentation.mp4](assets/videos/presentation.mp4) - Project presentation and demo

Outputs

- [assets/outputs/ner_visualization_simple.html](assets/outputs/ner_visualization_simple.html) - Simple NER visualization
- [assets/outputs/ner_visualization.html](assets/outputs/ner_visualization.html) - Enhanced NER visualization

## Notebooks

- [src/notebooks/task1_iris_classification_py.ipynb](src/notebooks/task1_iris_classification_py.ipynb) - Iris dataset EDA and decision tree classification
- [src/notebooks/task2_mnist_cnn_py.ipynb](src/notebooks/task2_mnist_cnn_py.ipynb) - CNN implementation for MNIST digit recognition
- [src/notebooks/task3_spacy_nlp.ipynb](src/notebooks/task3_spacy_nlp.ipynb) - NLP analysis with spaCy and sentiment detection

## How to run

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
## How to run

1. Create a virtual environment and install dependencies:

``` terminal
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
streamlit run app.py
Access the web interface at http://localhost:8501

```

