from keras.models import load_model
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
import cv2
import glob
# from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.utils.multiclass import unique_labels

# image = cv2.imread('/home/duongnh/DATN/test_emotion/challenges-in-representation-learning-facial-expression-recognition-challenge/635s_fer2013/test/4/1.jpg', 0)
# image = cv2.resize(image, (48, 48))
path_model_test = '/home/duongnh/DATN/test_emotion/challenges-in-representation-learning-facial-expression-recognition-challenge/m2_opeset_1_emotion_clf_v1.h5'
model = load_model(path_model_test)

# image = np.reshape(image, [1, 48, 48, 1])
# print(image)
# result  = model.predict(image)
# result  = model.predict_classes(image)
path_data_test = '/home/duongnh/DATN/test_emotion/open_set_1/test1'

# format_image = 'jpg'

# print (result)
pred = []
for filename in glob.glob(path_data_test + '/0/*'):
	image = cv2.imread(filename, 0)
	image = cv2.resize(image, (48, 48))
	image = np.reshape(image, [1,48,48,1])
	result = model.predict_classes(image)
	pred.append(result)

for filename in glob.glob(path_data_test + '/1/*'):
	image = cv2.imread(filename, 0)
	image = cv2.resize(image, (48, 48))
	image = np.reshape(image, [1,48,48,1])
	result = model.predict_classes(image)
	pred.append(result)

for filename in glob.glob(path_data_test + '/2/*'):
	image = cv2.imread(filename, 0)
	image = cv2.resize(image, (48, 48))
	image = np.reshape(image, [1,48,48,1])
	result = model.predict_classes(image)
	pred.append(result)

for filename in glob.glob(path_data_test + '/3/*'):
	image = cv2.imread(filename, 0)
	image = cv2.resize(image, (48, 48))
	image = np.reshape(image, [1,48,48,1])
	result = model.predict_classes(image)
	pred.append(result)

for filename in glob.glob(path_data_test + '/4/*'):
	image = cv2.imread(filename, 0)
	image = cv2.resize(image, (48, 48))
	image = np.reshape(image, [1,48,48,1])
	result = model.predict_classes(image)
	pred.append(result)

for filename in glob.glob(path_data_test + '/5/*'):
	image = cv2.imread(filename, 0)
	image = cv2.resize(image, (48, 48))
	image = np.reshape(image, [1,48,48,1])
	result = model.predict_classes(image)
	pred.append(result)

for filename in glob.glob(path_data_test + '/6/*'):
	image = cv2.imread(filename, 0)
	image = cv2.resize(image, (48, 48))
	image = np.reshape(image, [1,48,48,1])
	result = model.predict_classes(image)
	pred.append(result)

# 	if result == [0]:
# 		sum += 1
# 	print(result)
pred = np.array(pred)
pred = pred.ravel()
print(np.shape(pred))
number_img = len(pred)

test_batches = ImageDataGenerator().flow_from_directory(path_data_test, target_size = (48, 48), color_mode = 'grayscale', batch_size = 50, shuffle = False, seed = 2)
# print(test_batches.class_indices)

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# labels = ['Fear', 'Sad', 'Disgust', 'Surprise', 'Happy', 'Neutral', 'Angry']
labels = np.array(labels)
# print(type(labels))
# title='Confusion matrix'

# model = load_model('635s_emotion_clf_v1.h5')

# test_batches.reset()

loss , acc = model.evaluate_generator(test_batches, number_img//50+1)

print(acc)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(np.array(test_batches.classes, 'int'), pred, classes=labels, normalize=True,
                      title='Normalized confusion matrix')

# plt.show()

# plt.savefig('/home/duongnh/Pictures/testsave.png')

# fig1 = plt.gcf()
plt.show()
# plt.draw()
# fig1.savefig('/home/duongnh/Pictures/m2_fer_2013_privateTest.png', dpi=100)

