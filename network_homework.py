import streamlit as st
from utils import PrepProcesor, columns

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow
import matplotlib.pyplot  as plt


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.load_weights('weights.h5')

def main(model=model):
    st.title('Computer Networks and Data Communication Homework')
    st.write('Upload dog or cat image')

    file = st.file_uploader('Please upload an image', type=['jpg', 'png'])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        resized_image = image.resize((128, 128))
        img_array = np.array(resized_image) / 255
        img_array = img_array.reshape(1, 128, 128, 3)

        predictions = model.predict(img_array)
        classes = ['cat', 'dog']

        fig, ax = plt.subplots()
        y_pos = np.arange(len(classes))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title('Classes')

        st.pyplot(fig)
    else:
        st.text('You have not uploaded an image yet.')


if __name__ == '__main__':
    main()