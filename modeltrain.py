import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import scipy

# Define the path to your preprocessed dataset
train_dataset_path = 'D:\\MINI PROJECT\\my_dataset\\train\\'
test_dataset_path = 'D:\\MINI PROJECT\\my_dataset\\test\\'

# Data preprocessing (rescaling)
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load and preprocess the training dataset
train_generator = train_datagen.flow_from_directory(
    train_dataset_path,
    target_size=(224, 224),
    batch_size=32,  # Adjust batch size as needed
    class_mode='categorical')

# Load and preprocess the testing dataset
test_generator = test_datagen.flow_from_directory(
    test_dataset_path,
    target_size=(224, 224),
    batch_size=32,  # Adjust batch size as needed
    class_mode='categorical')

# Create a DNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Optional dropout layer
model.add(Dense(4, activation='softmax'))  # Assuming 4 classes

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // train_generator.batch_size,
                    epochs=10,  # Adjust the number of epochs as needed
                    validation_data=test_generator,
                    validation_steps=test_generator.samples // test_generator.batch_size,
                    verbose=1)

# Save the trained model
model.save('alzheimers_model.keras')
