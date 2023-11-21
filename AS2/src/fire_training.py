#!/usr/bin/env python
# coding: utf-8

# Import the library to process
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
from imutils import paths
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import confusion_matrix
import itertools

file_location = os.path.abspath(__file__)
root_directory = os.path.dirname(file_location)
image_size = 224
data_path = os.path.join(root_directory, '..', "fire_dataset/")
# Stage 02: Data Preprocessing \
# This stage convert the images are selected in disk, performing various stage such as resize image, one hot encoding on labels, convert image size

#Take list of train image
train_image_path = list(paths.list_images(data_path + "train"))
#Take list of val image
val_image_path = list(paths.list_images(data_path + "test"))
#Make list of class: wear_mask and unwear_mask
classNames = np.array(sorted(os.listdir(data_path + "train")))
#Build function to process images
def preprocess_image(image_path):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_image(image,channels=3,expand_animations=False)
  image = tf.image.resize(image,(image_size,image_size))
  image = image / 255.0

  label = tf.strings.split(image_path,os.path.sep)[-2]
  oneHot = label == classNames
  encodedLabel = tf.argmax(oneHot)

  return (image,encodedLabel)


#Using Tf.data to create pipeline load and process image
train_dataset = tf.data.Dataset.from_tensor_slices(train_image_path)
train_dataset = (train_dataset
                 .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(64)
                 .prefetch(tf.data.AUTOTUNE)
                 )
val_dataset = tf.data.Dataset.from_tensor_slices(val_image_path)
val_dataset = (val_dataset
                 .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(64)
                 .prefetch(tf.data.AUTOTUNE)
                 )


img_fires = []
labels_fires = []
fire_path = data_path + "train/" + "fire_images/"
nonFire_path = data_path + "train/" + "non_fire_images/"
for i in os.listdir(fire_path):
  img = os.path.join(fire_path + "/" ,i)
  img_fires.append(img)
  labels_fires.append("fire")
img_no_fires = []
target_no_fires = []
for i in os.listdir(nonFire_path):
  img = os.path.join(nonFire_path + "/" ,i)
  img_no_fires.append(img)
  target_no_fires.append("non fire")
fire_df = pd.DataFrame()
fire_df["image"] = img_fires
fire_df["target"] = labels_fires
no_fires_df = pd.DataFrame()
no_fires_df["image"] = img_no_fires
no_fires_df["target"] = target_no_fires
df = pd.concat([fire_df, no_fires_df], axis = 0,ignore_index = True)
df = shuffle(df)
print(df)

plt.figure(figsize = (12,8))
img = load_img(no_fires_df["image"][10])
plt.imshow(img)
plt.title("Non Fire",color="green",size = 14)
plt.grid(color='#999999',linestyle='-')
plt.show()

x_train = df.iloc[0:10000, ]
x_val = df.iloc[10001:11001, ]
x_test = df.iloc[11002:, ]


print("Train Seti:","\n",x_train["target"].value_counts(),"\n""Validation Seti: ","\n", x_val["target"].value_counts(), "\n"
      "Test Seti: ", "\n",x_test["target"].value_counts())

model_dir = os.path.join(root_directory, '..', 'saved_model/')
model_h5_path = os.path.join(model_dir, 'model.h5')
model_h5_weight = os.path.join(model_dir, "model_emotion.h5")
checkpoint = ModelCheckpoint(model_h5_path,monitor='val_acc',verbose = 1,save_best_only=True,mode='max')
model = load_model(model_h5_path)
if os.path.exists(model_h5_weight):
    model.load_weights(model_h5_weight)
#Training model
history = model.fit(
    train_dataset,
    steps_per_epoch= len(train_dataset),
    epochs = 50,
    validation_data = val_dataset,
    validation_steps = len(val_dataset),
    verbose = 1
)

model.save_weights(model_h5_weight)
print("Saved Successful")

acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc))

plt.figure(1)
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend( ["Training Accuracy", "Validation Accuracy"])
plt.title ('Training and validation accuracy')
plt.figure(2)
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend( ["Training Loss", "Validation Loss"])
plt.title ('Training and validation loss'   )


# Plot Confusion Matrix - Examine Accuracy, loss of model are trained
train_dir = data_path + 'train'
val_dir = data_path + 'test'
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
val_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
  plt.imshow(cm,interpolation='nearest',cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks,classes,rotation=45)
  plt.yticks(tick_marks,classes)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis = 1)[:,np.newaxis]
    print("Normalized confusion matrix")
  else:
    print("Confusion matrix, without normalization")
  print(cm)

  thresh = cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i, cm[i,j],horizontalalignment="center",color="white" if cm[i,j] > thresh else "black")

y_pred = model.predict(val_dataset,verbose=0)
predict = []
for i in range(len(y_pred)):
  if y_pred[i][0] > 0.5:
    predict.append(1)
  else:
    predict.append(0)


x_pred = model.predict(train_dataset,verbose=0)
predict_train = []
for i in range(len(x_pred)):
  if x_pred[i][0] > 0.5:
    predict_train.append(1)
  else:
    predict_train.append(0)
cm = confusion_matrix(validation_generator.classes,predict)
plot_confusion_matrix(cm = cm,classes=classNames,title='Confusion Matrix Testing')
cm_train = confusion_matrix(train_generator.classes,predict_train)
plot_confusion_matrix(cm = cm_train, classes=classNames,title='Confusion Matrix Training')

