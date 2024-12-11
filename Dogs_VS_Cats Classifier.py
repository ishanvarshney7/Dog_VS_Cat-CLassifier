!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d salader/dogs-vs-cats

import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# prompt: adddata augumentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '/content/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '/content/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

train_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/train',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (256, 256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/test',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (256, 256)
)

def process(image, label):
  image = tf.cast(image/255. , tf.float32)
  return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))


model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))


model.add(Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))


model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_ds, epochs=10, validation_data=validation_ds)

import io
import ipywidgets as widgets
from IPython.display import display
from PIL import Image
import numpy as np
import tensorflow as tf

# Upload image widget
uploader = widgets.FileUpload(accept='image/*', multiple=False)
display(uploader)

# Button to trigger prediction
predict_button = widgets.Button(description="Predict")
display(predict_button)

# Output area for prediction
output = widgets.Output()
display(output)

def on_predict_button_clicked(b):
    output.clear_output()  # Clear previous output
    if not uploader.value:
        with output:
            print("No file uploaded. Please upload an image.")
        return

    try:
        # Get uploaded image
        uploaded_image = list(uploader.value.values())[0]
        image_bytes = uploaded_image['content']

        # Open image using PIL
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":  # Ensure image is in RGB mode
            image = image.convert("RGB")
        image = image.resize((256, 256))  # Resize the image to 256x256
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(image_array)

        # Process prediction
        predicted_class = "Dog" if prediction[0][0] > 0.5 else "Cat"
        probability = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        with output:
            print(f"Predicted Class: {predicted_class}")
            print(f"Probability: {probability:.2f}")
    except Exception as e:
        with output:
            print(f"An error occurred: {e}")

predict_button.on_click(on_predict_button_clicked)