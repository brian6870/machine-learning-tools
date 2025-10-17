
# task2_mnist_cnn.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class MNISTClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_and_preprocess_data(self):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset...")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()

        print(f"Training data shape: {self.x_train.shape}")
        print(f"Training labels shape: {self.y_train.shape}")
        print(f"Test data shape: {self.x_test.shape}")
        print(f"Test labels shape: {self.y_test.shape}")

        # Normalize pixel values
        self.x_train = self.x_train.astype("float32") / 255.0
        self.x_test = self.x_test.astype("float32") / 255.0

        # Reshape data for CNN (add channel dimension)
        self.x_train = self.x_train.reshape(-1, 28, 28, 1)
        self.x_test = self.x_test.reshape(-1, 28, 28, 1)

        print(f"After reshaping - Training data shape: {self.x_train.shape}")
        print(f"After reshaping - Test data shape: {self.x_test.shape}")

        return self.x_train, self.y_train, self.x_test, self.y_test

    def explore_dataset(self):
        """Explore the MNIST dataset"""
        print("\n=== MNIST Dataset Exploration ===")
        print(f"Unique labels: {np.unique(self.y_train)}")
        print(f"Label distribution in training set: {np.bincount(self.y_train)}")
        print(f"Label distribution in test set: {np.bincount(self.y_test)}")

        # Display sample images
        self.display_sample_images()

    def display_sample_images(self, num_samples=10):
        """Display sample images from each class"""
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i in range(10):  # 0-9 digits
            # Find first occurrence of each digit
            idx = np.where(self.y_train == i)[0][0]
            axes[i].imshow(self.x_train[idx].reshape(28, 28), cmap='gray')
            axes[i].set_title(f'Digit: {i}')
            axes[i].axis('off')

        plt.suptitle('Sample Images from MNIST Dataset', fontsize=16)
        plt.tight_layout()
        plt.savefig('../../assets/images/mnist_sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()

    def build_cnn_model(self):
        """Build CNN model architecture"""
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print("Model architecture:")
        model.summary()

        return model

    def train_model(self, epochs=15, batch_size=128):
        """Train the CNN model"""
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7
            )
        ]

        # Train model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('../../assets/images/mnist_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_model(self):
        """Evaluate model performance"""
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"\n=== Model Evaluation ===")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Predictions
        y_pred_proba = self.model.predict(self.x_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('../../assets/images/mnist_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        return test_accuracy, y_pred

    def visualize_predictions(self, num_samples=10):
        """Visualize model predictions on test samples"""
        # Get predictions
        y_pred_proba = self.model.predict(self.x_test[:num_samples])
        y_pred = np.argmax(y_pred_proba, axis=1)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i in range(num_samples):
            axes[i].imshow(self.x_test[i].reshape(28, 28), cmap='gray')
            true_label = self.y_test[i]
            pred_label = y_pred[i]
            confidence = np.max(y_pred_proba[i])

            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.2f}',
                            color=color, fontsize=10)
            axes[i].axis('off')

        plt.suptitle('Model Predictions on Test Samples', fontsize=16)
        plt.tight_layout()
        plt.savefig('../../assets/images/mnist_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_complete_analysis(self):
        """Run complete MNIST analysis pipeline"""
        # Load and preprocess data
        self.load_and_preprocess_data()

        # Explore dataset
        self.explore_dataset()

        # Build model
        self.build_cnn_model()

        # Train model
        self.train_model(epochs=15)

        # Plot training history
        self.plot_training_history()

        # Evaluate model
        test_accuracy, y_pred = self.evaluate_model()

        # Visualize predictions
        self.visualize_predictions()

        # Save model
        self.model.save('../models/mnist_cnn_model.h5')
        print("Model saved as '../models/mnist_cnn_model.h5'")

        return test_accuracy

if __name__ == "__main__":
    # Run complete MNIST classification
    mnist_classifier = MNISTClassifier()
    accuracy = mnist_classifier.run_complete_analysis()
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")