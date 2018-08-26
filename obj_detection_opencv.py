
# Reference: https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

# import the necessary packages
import numpy as np
import cv2

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

confidence_threshold = 0.2
blob_scale_factor = 0.007843
blob_mean = 127.5

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('/Users/vincent/PycharmProjects/test/MobileNetSSD_deploy.prototxt', '/Users/vincent/PycharmProjects/test/MobileNetSSD_deploy.caffemodel')

cap = cv2.VideoCapture(0)

while(cap.isOpened()):

    ret, frame = cap.read()

    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)

    (h, w) = frame.shape[:2]

    # MobileNet requires fixed dimensions for input image(s)
    # so we have to ensure that it is resized to 300x300 pixels.
    # set a scale factor to image because network the objects has differents size.
    # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
    #
    #   blob = (frame - mean)/scalefactor
    #
    # after executing this command our "blob" now has the shape:
    # (1, 3, 300, 300)
    blob = cv2.dnn.blobFromImage(image=cv2.resize(frame, (300, 300)), scalefactor=blob_scale_factor, mean=blob_mean)

    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2, lineType=cv2.LINE_AA)
            y = startY - 15 if startY - 15 > 15 else startY + 15

            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[idx], thickness=1, lineType=cv2.LINE_AA)

    # show the output image
    cv2.imshow("Output", frame)
    cv2.waitKey(25)