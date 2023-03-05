#kidus berhanu
from keras.applications import VGG16
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Model

# function to Load the VGG16 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


    #  calling function to  Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    #  to Add extra convolutional layers
    x = base_model.output
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    #  to Flatten the output and add a fully connected layer
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)

    # Add a softmax classifier for fruit classification
    output = Dense(num_classes, activation='softmax')(x)

    #  to Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    #  for Compilling  the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, X_train, y_train):
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

def evaluate_model(model, X_test, y_test):
    # evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])
    
def predict(model, X_predict):
    # make predictions
    predictions = model.predict(X_predict)
    return predictions

num_classes = 5
model = create_model(num_classes)
train_model(model, X_train, y_train)
evaluate_model(model, X_test, y_test)
predictions = predict(model, X_predict)
