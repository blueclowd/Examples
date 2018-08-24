
# Reference: https://www.pyimagesearch.com/2016/08/22/an-intro-to-linear-classification-with-python/

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np
import cv2
import os


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


# grab the list of images that we'll be describing
print("[INFO] describing images...")

image_folder = 'dataset'
imagePaths = os.listdir('dataset')

# initialize the data matrix and labels list
data = []
labels = []

# loop over the input images
for imagePath in imagePaths:

    if not imagePath.endswith(('jpg', 'bmp', 'png', 'jpeg')):

        continue

    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(os.path.join(image_folder, imagePath))
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    print(image.shape)

    # extract a color histogram from the image, then update the
    # data matrix and labels list
    hist = extract_color_histogram(image)
    data.append(hist)
    labels.append(label)

    # show an update every 1,000 images

    print("[INFO] processed {}".format(label))


# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
    np.array(data), labels, test_size=0.25, random_state=42)

# train the linear regression clasifier
print("[INFO] training Linear SVM classifier...")
model = LinearSVC()
model.fit(trainData, trainLabels)

# evaluate the classifier
print("[INFO] evaluating classifier...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions,
                            target_names=le.classes_))

joblib.dump(model, 'linearSVCModel.pkl')