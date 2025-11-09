import tensorflow as tf

# Define image dimensions and batch size (you might need to define these if not already done)
# img_height = 224
# img_width = 224
# batch_size = 32

# Specify the directory where the dataset is located
data_dir = '/content/deforestation_images/' # Replace with the actual path to your dataset directory

# Load the dataset from the specified directory
try:
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'  # Or 'int' depending on your label format
    )
    print("Dataset loaded successfully.")
    print(f"Class names: {raw_ds.class_names}")
    print(f"Number of batches: {tf.data.experimental.cardinality(raw_ds).numpy()}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    print(f"Please ensure the dataset is uploaded to the specified path: {data_dir}")

from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Get the number of classes from the training dataset
num_classes = len(train_ds.class_names)

# Load the pre-trained MobileNetV3Small model
# Setting include_top=False removes the classification layer
base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add a global average pooling layer
predictions = Dense(num_classes, activation='softmax')(x) # Add a dense layer with softmax activation for classification

# Create the new model
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()
print(f"MobileNetV3 model built with {num_classes} output classes.")

import tensorflow as tf
import numpy as np

# Define image dimensions and batch size (assuming these were defined previously)
# img_height = 224
# img_width = 224
# batch_size = 32

# Specify the directory where the dataset is located
data_dir = '/content/deforestation_images/' # Update this path to where you upload the dataset

# Load the dataset from the specified directory
try:
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'  # Assuming categorical labels for classification
    )

    # Define the preprocessing function
    def preprocess(image, label):
        # Normalize pixel values to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Apply preprocessing to the dataset
    preprocessed_ds = raw_ds.map(preprocess)

    # Get the number of batches in the dataset
    DATASET_SIZE = tf.data.experimental.cardinality(preprocessed_ds).numpy()
    train_size = int(0.8 * DATASET_SIZE)
    val_size = DATASET_SIZE - train_size

    # Split the dataset into training and validation sets
    train_ds = preprocessed_ds.take(train_size)
    val_ds = preprocessed_ds.skip(train_size).take(val_size)

    # Configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("Dataset loaded and preprocessed successfully.")
    print(f"Training batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}")

except Exception as e:
    print(f"Error loading or preprocessing dataset: {e}")
    print(f"Please ensure the dataset is uploaded to the specified path: {data_dir}")

import tensorflow as tf

# Define image dimensions and batch size
img_height = 224
img_width = 224
batch_size = 32

# Load the dataset from the specified directory
# Assuming the dataset is in a directory named 'deforestation_images' in the current working directory
data_dir = 'deforestation_images'

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Define a function to preprocess the images
def preprocess_image(image, label):
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return image, label

# Apply the preprocessing function to both datasets
train_ds = train_ds.map(preprocess_image)
val_ds = val_ds.map(preprocess_image)

# Configure the datasets for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Training dataset loaded and preprocessed.")
print("Validation dataset loaded and preprocessed.")

import tensorflow as tf

dataset_dir = '/kaggle/input/forest-aerial-deforestation-images/deforestation_images'

# Load the dataset
raw_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'  # Assuming categorical labels for classification
)

def preprocess(image, label):
    # Normalize pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply preprocessing to the dataset
preprocessed_ds = raw_ds.map(preprocess)

# Get the number of batches in the dataset
DATASET_SIZE = tf.data.experimental.cardinality(preprocessed_ds).numpy()
train_size = int(0.8 * DATASET_SIZE)
val_size = DATASET_SIZE - train_size

# Split the dataset into training and validation sets
train_ds = preprocessed_ds.take(train_size)
val_ds = preprocessed_ds.skip(train_size).take(val_size)

# Configure datasets for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Dataset loaded and preprocessed successfully.")
print(f"Training batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"Validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}")

import os

# List files in the Kaggle input directory to find the dataset location
print(os.listdir('/kaggle/input/'))

import os

# List files in the root directory to find the dataset location
print(os.listdir('/'))

import os

# List files in the /kaggle directory to find potential dataset locations
print(os.listdir('/kaggle/'))

import zipfile
import os

zip_path = '/content/archive.zip'
extract_dir = '/content/deforestation_dataset'

# Create the extraction directory if it doesn't exist
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Open the zip file and extract its contents
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Dataset extracted to: {extract_dir}")

import tensorflow as tf
import numpy as np

# Define the directory path where the dataset was extracted
dataset_dir = extract_dir

# Load the dataset
raw_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'  # Assuming categorical labels for classification
)

# Define the preprocessing function
def preprocess_image(image, label):
    # Normalize pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply preprocessing to the dataset
preprocessed_ds = raw_ds.map(preprocess_image)

# Determine the number of training and validation batches
dataset_size = tf.data.experimental.cardinality(preprocessed_ds).numpy()
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# Split the dataset into training and validation sets
train_ds = preprocessed_ds.take(train_size)
val_ds = preprocessed_ds.skip(train_size)

# Configure datasets for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Dataset loaded and preprocessed successfully.")
print(f"Training batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"Validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}")

from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Get the number of classes from the training dataset
num_classes = len(raw_ds.class_names)

# Load the pre-trained MobileNetV3Small model
# Setting include_top=False removes the classification layer
base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add a global average pooling layer
predictions = Dense(num_classes, activation='softmax')(x) # Add a dense layer with softmax activation for classification

# Create the new model
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()
print(f"MobileNetV3 model built with {num_classes} output classes.")

import tensorflow as tf

# Define the optimizer, loss function, and metrics
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()
metrics = ['accuracy']

# Compile the model
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=metrics)

# Train the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

print("Model training completed.")

# Evaluate the model on the validation dataset
loss, accuracy = model.evaluate(val_ds)

print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Take a batch of images and labels from the validation dataset
for images, labels in val_ds.take(1):
    batch_images = images
    batch_labels = labels

# Make predictions on the batch of images
predictions = model.predict(batch_images)

print("\nPredictions made on a batch of images from the validation dataset.")

import matplotlib.pyplot as plt
import numpy as np

# Get class names from the dataset
class_names = raw_ds.class_names

plt.figure(figsize=(10, 10))
for i in range(min(9, len(batch_images))):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(batch_images[i].numpy())

    # Get the true label
    true_label = np.argmax(batch_labels[i].numpy())
    predicted_label = np.argmax(predictions[i])

    title_color = "green" if true_label == predicted_label else "red"

    plt.title(f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}", color=title_color)
    plt.axis("off")
plt.tight_layout()
plt.show()