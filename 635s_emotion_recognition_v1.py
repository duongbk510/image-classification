from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import SGD
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

config = K.tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(K.tf.Session(config=config))

train_path = 'pre_fer2013/train'
validation_path = 'pre_fer2013/valid'
test_path = 'pre_fer2013/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size = (48, 48), color_mode = 'grayscale',batch_size = 50)
validation_batches = ImageDataGenerator().flow_from_directory(validation_path, target_size = (48, 48), color_mode ='grayscale', batch_size = 50)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size = (48, 48), color_mode = 'grayscale', batch_size = 50)

model = Sequential()
model.add(Conv2D(64, (5,5), input_shape = (48, 48, 1)))
model.add(Activation('relu'))
# batch norm
model.add(MaxPooling2D(pool_size = (3,3), strides =2))

model.add(Conv2D(64, (5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (3,3), strides =2))

model.add(Conv2D(128, (4,4)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(3072))
model.add(Activation('relu'))

model.add(Dense(7))
model.add(Activation('softmax'))

model.summary()

model.compile(optimizer = SGD(lr=0.001, momentum=0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit_generator(train_batches, steps_per_epoch = 226, validation_data = validation_batches, validation_steps = 30, epochs = 100, verbose = 2)

loss , acc = model.evaluate_generator(test_batches)

print(acc)

model.save('635s_emotion_clf_v1.h5')