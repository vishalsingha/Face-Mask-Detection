# import library
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# directory and categries
DIRECTORY = r"C:\Users\Vipul Singh\Desktop\dataset"
CATEGORIES = {"with_mask":1, "without_mask":0}

# loading data from folder
def load_data(category, path):
    '''category:dictationary of categories as the label and labels as the value. \n 
    path: the path of directory which have data.
    '''
    data = []
    labels = []
    for label in CATEGORIES:
        file_path = DIRECTORY+ "\\" +label
        for img in os.listdir(file_path):
            img_path = os.path.join(file_path, img)
            img = load_img(img_path, target_size = (224, 224))
            img = img_to_array(img)
            img = preprocess_input(img)
            data.append(img)
            labels.append(CATEGORIES[label])
    return data, labels

data, labels = load_data(CATEGORIES, DIRECTORY)

# generate one-hot vactor for labels
labels = to_categorical(labels)

# convert into array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# train and validation split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size  = 0.20)

# data augmentation
datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2,rotation_range=20, shear_range=0.15,horizontal_flip=True, fill_mode="nearest", zoom_range=0.15,)

# load basemodel(mobileNetv2)
baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

# adiing layers to base model for fine-tuning
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# creating model
model = Model(inputs=baseModel.input, outputs=headModel)

# freeze layers of basemodel
for layer in baseModel.layers:
    layer.trainable = False

EPOCHS = 20
BS = 32
INIT_LR = 1e-4

# creating optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile model
model.compile(loss="binary_crossentropy", optimizer=opt , metrics=["accuracy"])

# train model(fine-tuning by using data-augmentation)
History = model.fit(datagen.flow(x_train, y_train, batch_size=BS), steps_per_epoch=len(x_train) // BS, validation_data=(x_test, y_test), validation_steps=len(x_test) // BS, epochs=EPOCHS)

# predict the x_test
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

# print classification report
print(classification_report(y_test.argmax(axis=1), y_pred, target_names=CATEGORIES))

# save the model
model.save("mask_detector.model", save_format="h5")

# plot accuracy and loss
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), History.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), History.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), History.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), History.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("No of Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(3)
plt.savefig("plot.png")


