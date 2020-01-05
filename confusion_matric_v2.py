from keras.models import load_model
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.utils.multiclass import unique_labels

# iris = datasets.load_iris()
# print(type(iris.target_names))

# validation_path = 'fer2013/PublicTest'
# test_path = 'fer2013/PrivateTest'
# test_path = 'pre_jaffe'
test_path = 'pre_fer2013/test'

# validation_batches = ImageDataGenerator().flow_from_directory(validation_path, target_size = (48, 48), color_mode='grayscale', batch_size = 64)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size = (48, 48), color_mode = 'grayscale', batch_size = 50, shuffle = False, seed = 2)
print(test_batches.class_indices)

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# labels = ['Fear', 'Sad', 'Disgust', 'Surprise', 'Happy', 'Neutral', 'Angry']
labels = np.array(labels)
# print(type(labels))
# title='Confusion matrix'

model = load_model('635s_emotion_clf_v1.h5')

# test_batches.reset()

loss , acc = model.evaluate_generator(test_batches, 1343//50+1)

print(acc)

predict = model.predict_generator(test_batches, 1343//50+1)
# cacul = test_batches.classes[test_batches.index_array]

predict_y = np.argmax(predict, axis=1)
# print(sum(predict_y==cacul)/2726)
# I = predict_y.shape[0]
print(np.shape(predict_y))
print(np.shape(test_batches.classes))
# print(confusion_matrix(test_batches.classes[test_batches.index_array], predict_y))
# print(model.metrics_names)
# cnt = 0
# a = predict_y - test_batches.classes
# for i in a:
# 	if i != 0:
# 		cnt += 1
# print(type(predict_y), type(np.array(test_batches.classes, 'int')))

# predict_y = np.array(predict_y, 'int')
# print(predict_y.shape)
# cm = confusion_matrix(test_batches.classes, predict_y)
# # print(np.shape(test_batches.classes))
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# # print('confusion matric')
# # print(cm)

# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title(title)
# plt.colorbar()
# tick_marks = np.arange(len(labels))
# plt.xticks(tick_marks, labels, rotation=45)
# plt.yticks(tick_marks, labels)
# # fmt = 'd'
# fmt = '.2f'
# thresh = cm.max() / 2.
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(j, i, format(cm[i, j], fmt),
#             horizontalalignment="center",
#             color="white" if cm[i, j] > thresh else "black")

# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.tight_layout()
# plt.show()

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

plot_confusion_matrix(np.array(test_batches.classes, 'int'), predict_y, classes=labels, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
