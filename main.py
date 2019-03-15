from PIL import Image

import cv2
import os


def main():
	image = Image.open("image.png")
	image.save("temp.png")
	cascPath = "haarcascade_frontalface_default.xml"

	# Read the image
	image = cv2.imread("temp.png")
	if image is None:
		print("Invalid picture!")
		exit(-1)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Create the haar cascade
	faceCascade = cv2.CascadeClassifier(cascPath)
	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
	)

	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow("Faces found", image)
	cv2.waitKey(0)

	os.remove("temp.png")


if __name__ == "__main__":
	main()
