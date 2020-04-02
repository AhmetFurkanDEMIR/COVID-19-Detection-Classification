from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
"""

python test_network.py --model asd.h5 \
	--image resima_di.png

"""
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
orig = image.copy()
# pre-process the image for classification
image = cv2.resize(image, (150, 150))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


print("[INFO] loading network...")
model = load_model(args["model"])


if model.predict(image) >=0.5:

	label = "COVID-19"
	a = 100 * model.predict(image)[0][0]

else:

	label = "HEALTY"
	a = (1-model.predict(image)[0][0]) * 100


os.system("clear")


proba = model.predict(image)
label = "{}: {:.2f}%".format(label, a)

output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

cv2.imshow("Output", output)
cv2.waitKey(0)


