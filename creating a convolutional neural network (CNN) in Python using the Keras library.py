#KIDUS BERHANU
# Data configuration
training_set_folder = './fruits-360/Training_smaller'
test_set_folder     = './fruits-360/Test_smaller'

# Model configuration
batch_size = 25
img_width, img_height, img_num_channels = 25, 25, 3
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 25
optimizer = Adam()
verbosity = 1

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Create a generator
train_datagen = ImageDataGenerator(
  rescale=1./255
)
train_datagen = train_datagen.flow_from_directory(
        training_set_folder,
        save_to_dir='./adapted-images',
        save_format='jpeg',
        batch_size=batch_size,
        target_size=(25, 25),
        class_mode='sparse')

# Define model architecture
model = Sequential()
model.add(InputLayer(input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Compile model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])

# Train model
model.fit(
        train_datagen,
        epochs=no_epochs,
        verbose=verbosity)
        

# Evaluate model on test data
_, acc = model.evaluate(test_datagen, verbose=0)
print('> %.3f' % (acc * 100.0))
def load_data(training_set_folder, test_set_folder):
    # Create a generator for the training set
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        training_set_folder,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='sparse')
    
    # Create a generator for the test set
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_set_folder,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='sparse')
    
    return train_generator, test_generator

# Load the data
train_generator, test_generator = load_data(training_set_folder, test_set_folder)

def train_and_evaluate(model, train_generator, test_generator, no_epochs):
    # Train model
    model.fit(
        train_generator,
        epochs=no_epochs,
        verbose=verbosity)

    # Evaluate model on test data
    _, acc = model.evaluate(test_generator, verbose=0)
    print('> %.3f' % (acc * 100.0))

# Train and evaluate the model
train_and_evaluate(model, train_generator, test_generator, no_epochs)


def save_model(model, model_file):
    model.save(model_file)

# Save the model to a file
save_model(model, 'model.h5')


def load_model(model_file):
    return load_model(model_file)

# Load a saved model
model = load_model('model.h5')


def predict(model, data):
    return model.predict(data)

# Make a prediction on new data
predictions = predict(model, new_data)


