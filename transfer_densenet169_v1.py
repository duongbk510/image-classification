from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import numpy as np
from keras import applications
from sklearn.metrics import confusion_matrix

config = K.tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(K.tf.Session(config=config))

train_path = 'close_set_1/train'
validation_path = 'close_set_1/test'
# test_path = 'fer2013/PrivateTest'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size = (200, 200), color_mode = 'rgb',batch_size = 64)
validation_batches = ImageDataGenerator().flow_from_directory(validation_path, target_size = (200, 200), color_mode ='rgb', batch_size = 64)
# test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size = (48, 48), color_mode = 'rgb', batch_size = 32)

conv_base = applications.densenet.DenseNet169(weights='imagenet', include_top=False, input_shape=(200,200,3))
conv_base.summary()

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(7))
model.add(Activation('softmax'))

conv_base.trainable = False

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-3, decay=1e-5, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch = 720, validation_data = validation_batches, validation_steps = 184, epochs = 25, verbose = 2)
model.save('densenet169_emotion_clf_v1.h5')

densenet169_close_json_v1 = model.to_json()
with open('densenet169_close_v1.json', 'w') as json_file:
	json_file.write(densenet169_close_json_v1)
model.save_weights('densenet169_close_v1_weights.h5')

loss , acc = model.evaluate_generator(validation_batches)

print(acc)