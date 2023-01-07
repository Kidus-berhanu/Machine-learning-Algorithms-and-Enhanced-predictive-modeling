# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the fruit dataset
X = np.load("fruit_images.npy")
y = np.load("fruit_labels.npy")

# Preprocess the data by rescaling the images and converting the labels to one-hot encoding
X = X / 255.0
y = keras.utils.to_categorical(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build the model using a CNN architecture
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(100, 100, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# Compile the model with an Adam optimizer and categorical cross-entropy loss
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model on the training data
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)


# Function to predict the class of a single fruit image
def predict_fruit(model, image):
  # Preprocess the image in the same way as the training data
  image = image / 255.0

  # Reshape the image to match the model's input shape
  image = image.reshape(1, 100, 100, 3)

  # Use the model to predict the class probabilities
  probabilities = model.predict(image)

  # Get the class label with the highest probability
  label = np.argmax(probabilities)

  return label

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred):
  # Convert the one-hot encoded labels back to integers
  y_true = np.argmax(y_true, axis=1)
  y_pred = np.argmax(y_pred, axis=1)

  # Calculate the confusion matrix
  confusion = tf.math.confusion_matrix(y_true, y_pred)

  # Normalize the confusion matrix
  confusion = confusion / np.sum(confusion, axis=1, keepdims=True)

  # Plot the confusion matrix using matplotlib
  import matplotlib.pyplot as plt
  plt.imshow(confusion, cmap="Blues")
  plt.colorbar()
  plt.xlabel("Predicted label")
  plt.ylabel("True label")
  plt.show()

# Function to display a fruit image and its predicted class
def display_prediction(model, image):
  # Get the predicted class label
  label = predict_fruit(model, image)

  # Convert the label back to a class name
  class_names = ["apple", "banana", "orange", "pear", "strawberry"]
  class_name = class_names[label]

  # Display the image and the predicted class name
  import matplotlib.pyplot as plt
  plt.imshow(image)
  plt.title(class_name)
  plt.show()
  
  
  # Function to save the model
def save_model(model, filename):
  # Serialize the model to a JSON file
  model_json = model.to_json()
  with open(filename + ".json", "w") as json_file:
    json_file.write(model_json)

  # Serialize the model weights to a HDF5 file
  model.save_weights(filename + ".h5")
  print("Model saved to", filename + ".json and", filename + ".h5")

# Function to load a saved model
def load_model(filename):
  # Load the model from the JSON file
  with open(filename + ".json", "r") as json_file:
    model_json = json_file.read()
  model = keras.models.model_from_json(model_json)

  # Load the model weights from the HDF5 file
  model.load_weights(filename + ".h5")
  print("Model loaded from", filename + ".json and", filename + ".h5")

  return model

# Function to generate a random batch of training data
def generate_batch(X, y, batch_size):
  # Create a random index into the training data
  index = np.random.randint(0, len(X), batch_size)

  # Select the corresponding images and labels
  X_batch = X[index]
  y_batch = y[index]

  return X_batch, y_batch

# Function to perform data augmentation on a batch of images
def augment_batch(X_batch, y_batch):
  # Initialize a list to store the augmented images and labels
  X_augmented = []
  y_augmented = []

  # Loop over the images and labels in the batch
  for X, y in zip(X_batch, y_batch):
    # Randomly flip the image horizontally
    if np.random.rand() < 0.5:
      X = np.fliplr(X)

    # Randomly rotate the image by up to 30 degrees
    angle = np.random.uniform(-30.0, 30.0)
    X = rotate(X, angle, reshape=False)

    # Randomly shift the image horizontally and vertically
    shift = np.random.uniform(-5.0, 5.0)
    tform = AffineTransform(translation=shift)
    X = warp(X, tform, mode="wrap")

    # Add the augmented image and label to the list
    X_augmented.append(X)
    y_augmented.append(y)

  # Convert the list to a NumPy array
  X_augmented = np.array(X_augmented)
  y_augmented = np.array(y_augmented)

  return X_augmented, y_augmented
  
  
  
  # Function to plot the training and validation loss
def plot_loss(history):
  # Extract the loss values from the history object
  train_loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  # Plot the loss values
  plt.plot(train_loss, label="Training loss")
  plt.plot(val_loss, label="Validation loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()

# Function to plot the training and validation accuracy
def plot_accuracy(history):
  # Extract the accuracy values from the history object
  train_acc = history.history["accuracy"]
  val_acc = history.history["val_accuracy"]

  # Plot the accuracy values
  plt.plot(train_acc, label="Training accuracy")
  plt.plot(val_acc, label="Validation accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()

# Function to fine-tune the model on a smaller dataset
def fine_tune(model, X_train, y_train, X_val, y_val, num_epochs):
  # Freeze the base model layers
  for layer in model.layers[:-4]:
    layer.trainable = False

  # Compile the model with an Adam optimizer and a lower learning rate
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss="categorical_crossentropy",
                metrics=["accuracy"])

  # Create a callback to reduce the learning rate if the validation loss plateaus
  lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                  factor=0.5,
                                                  patience=10,
                                                  min_lr=1e-5)

  # Create a callback to stop training if the validation loss does not improve
  early_stopper = keras.callbacks.EarlyStopping(monitor="val_loss",
# Function to implement the Adaptive Gradient Clipping (AGC) algorithm
def AGC(model, X_train, y_train, X_val, y_val, num_epochs, batch_size):
  # Compute the gradient norm for each layer
  grad_norms = [tf.norm(tensor) for tensor in model.optimizer.get_gradients(model.total_loss, model.trainable_variables)]

  # Define the AGC loss as the sum of the layer gradients
  agc_loss = tf.reduce_sum(grad_norms)

  # Define a custom Adam optimizer with the AGC loss as the objective function
  optimizer = keras.optimizers.Adam(learning_rate=1e-3)
  agc_update = optimizer.minimize(agc_loss, var_list=model.trainable_variables)

  # Define a custom training loop to perform AGC at each iteration
  @tf.function
  def train_step(inputs, labels):
    with tf.GradientTape() as tape:
      logits = model(inputs, training=True)
      loss_value = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    # Compute the gradients of the AGC loss with respect to the model parameters
    grads = tape.gradient(loss_value, model.trainable_variables)
    grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in grads]))

    # Update the model parameters using the AGC update rule
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    agc_update.run()

    # Return the loss value and the gradient norm
    return loss_value, grad_norm

  # Define a custom evaluation loop to compute the AGC loss
  @tf.function
  def val_step(inputs, labels):
    logits = model(inputs, training=False)
    loss_value = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return loss_value, agc_loss

  # Create a Metric object to track the AGC loss during training
  agc_loss_metric = keras.metrics.Mean(name="agc_loss")

  # Create a callback to record the AGC loss after each epoch
  class AGCCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs



