import numpy as np
import pandas as pd
import os
# import cv2
import matplotlib.pyplot as plt
# from tqdm import tqdm_notebook as tqdm
from sklearn.utils import class_weight, shuffle

from keras import applications
from keras import optimizers
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense
from keras.layers.core import Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


from google.colab import drive
# drive.mount('/content/drive')
foldernames = os.listdir('/milfotos/')
categories = []
files = []
i = 0
for k, folder in enumerate(foldernames):
    filenames = os.listdir("/milfotos/" + folder)
    for file in filenames:
        files.append("/milfotos/" + folder + "/" + file)
        categories.append(k)
        
df = pd.DataFrame({
    'filename': files,
    'category': categories
})
train_df = pd.DataFrame(columns=['filename', 'category'])
for i in range(10):
    train_df = train_df.append(df[df.category == i])

train_df.head()
train_df = train_df.reset_index(drop=True)

y = train_df['category']
x = train_df['filename']

x, y = shuffle(x, y, random_state=8)
images = []
with tqdm(total=len(train_df)) as pbar:
    for i, file_path in enumerate(train_df.filename.values):
        #read image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(150,150))
        images.append(img)
        pbar.update(1)

images = np.array(images)
rows,cols = 2,5
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20,20))
for i in range(10):
    path = train_df[train_df.category == i].values[6]
#     image = cv2.imread(path[0])/
    axes[i//cols, i%cols].set_title(path[0].split('/')[-2] + str(path[1]))
    axes[i//cols, i%cols].imshow(images[train_df[train_df.filename == path[0]].index[0]])
data_num = len(y)
random_index = np.random.permutation(data_num)

x_shuffle = []
y_shuffle = []
for i in range(data_num):
    x_shuffle.append(images[random_index[i]])
    y_shuffle.append(y[random_index[i]])
    
x = np.array(x_shuffle) 
y = np.array(y_shuffle)
val_split_num = int(round(0.2*len(y)))
x_train = x[val_split_num:]
y_train = y[val_split_num:]
x_test = x[:val_split_num]
y_test = y[:val_split_num]

print('x_train', x_train.shape)
print('y_train', y_train.shape)
print('x_test', x_test.shape)
print('y_test', y_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

img_rows, img_cols, img_channel = 150, 150, 3
name_animal = []
for i in range(10):
    path = train_df[train_df.category == i].values[2]
    if path[0].split('/')[-2] == 'caballo':
        name_animal.append('caballo')
    elif path[0].split('/')[-2] == 'ardilla':
        name_animal.append('ardilla')
    elif path[0].split('/')[-2] == 'mariposa':
        name_animal.append('mariposa')
    elif path[0].split('/')[-2] == 'vaca':
        name_animal.append('vaca')
    elif path[0].split('/')[-2] == 'gato':
        name_animal.append('gato')
    elif path[0].split('/')[-2] == 'oveja':
        name_animal.append('oveja')
    elif path[0].split('/')[-2] == 'gallina':
        name_animal.append('gallina')
    elif path[0].split('/')[-2] == 'elefante':
        name_animal.append('elefante')
    elif path[0].split('/')[-2] == 'arana':
        name_animal.append('arana')
    elif path[0].split('/')[-2] == 'perro':
        name_animal.append('perro')
input_shape=(img_rows, img_cols, img_channel)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (4, 4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (4, 4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam",
              metrics=['accuracy'])

print(model.summary())

batch_size = 16
epochs = 25

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test)
)


model.save('final_model.h5')
def show_plots(history):
    """ Useful function to view plot of loss values & accuracies across the various epochs """
    loss_vals = history['loss']
    val_loss_vals = history['val_loss']
    epochs = range(1, len(history['acc'])+1)
    
    f, ax = plt.subplots(nrows=1,ncols=2,figsize=(16,4))
    
    # plot losses on ax[0]
    ax[0].plot(epochs, loss_vals, color='navy',marker='o', linestyle=' ', label='Training Loss')
    ax[0].plot(epochs, val_loss_vals, color='firebrick', marker='*', label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='best')
    ax[0].grid(True)
    
    # plot accuracies
    acc_vals = history['acc']
    val_acc_vals = history['val_acc']

    ax[1].plot(epochs, acc_vals, color='navy', marker='o', ls=' ', label='Training Accuracy')
    ax[1].plot(epochs, val_acc_vals, color='firebrick', marker='*', label='Validation Accuracy')
    ax[1].set_title('Training & Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend(loc='best')
    ax[1].grid(True)
    
    plt.show()
    plt.close()
    
    # delete locals from heap before exiting
    del loss_vals, val_loss_vals, epochs, acc_vals, val_acc_vals
print(show_plots(history.history))

test_images = []
j = 185 # change this to get different images
for i in range(6):
    if i==0 or i==2 or i==4:
      path = train_df[train_df.category == i].values[j]
      a = images[train_df[train_df.filename == path[0]].index[0]]
      img = np.array(a)
      img = img[:, :, ::-1].copy() 
      # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      # img = cv2.resize(img, dsize=(150,150))
      test_images.append(img)

test_images = np.array(test_images).reshape(-1,150,150,3)
something = model.predict(test_images)
animals = name_animal
i = 0
j=0
for pred in something:
  if j==0 or j==2 or j==4:
    path = train_df[train_df.category == i].values[2]
    plt.imshow(test_images[i])
    plt.show()
    print('Actual  :', animals[j])
    print('Predicciòn :', animals[np.where(pred.max() == pred)[0][0]])
    i += 1
    j+=2
