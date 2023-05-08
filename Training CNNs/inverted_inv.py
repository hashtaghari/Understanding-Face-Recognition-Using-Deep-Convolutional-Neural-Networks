import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model

# Define image dimensions and number of classes
img_width, img_height = 224, 224
num_classes = 2

# Create data generators
train_dir = 'inverted dataset'
test_dir = 'desired dataset'

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(
                                                        img_width, img_height),
                                                    batch_size=32,
                                                    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(
                                                      img_width, img_height),
                                                  batch_size=32,
                                                  class_mode='binary')

# Load pre-trained VGG16 model without top layer
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(img_width, img_height, 3))

# Add custom top layers to the model
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

# Train the model on inverted images dataset
model.fit(train_generator, epochs=10, batch_size=32)

# Save the weights of the trained model to an .h5 file
model.save_weights('inverted_model_weights.h5')

# Extract activations in the penultimate fully-connected layer of the model
penultimate_layer_model = Model(
    inputs=model.input, outputs=model.get_layer(index=-2).output)
activations = penultimate_layer_model.predict_generator(
    test_generator, steps=len(test_generator))

# Test the model on upright images
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(
                                                      img_width, img_height),
                                                  batch_size=32,
                                                  class_mode='binary',
                                                  shuffle=False)

predictions = model.predict_generator(
    test_generator, steps=len(test_generator))
accuracy = np.mean(
    np.equal(np.argmax(predictions, axis=-1), test_generator.classes))
print('Test accuracy:', accuracy)
