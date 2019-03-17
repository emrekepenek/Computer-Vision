# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import  math
import glob
import matplotlib.pyplot as plt
import pandas as pd

def dist(p1,p2):
	x_dist=p1[0]-p2[0]
	y_dist=p1[1]-p2[1]
	dist= math.sqrt(x_dist**2+y_dist**2)
	return dist

def makAttr(shape):
	cene = shape[:17]
	sagkas = shape[17:22]
	solkas = shape[22:27]
	burun = shape[27:36]
	saggoz = shape[36:42]
	solgoz = shape[42:48]
	agiz = shape[48:67]

	ickasAragligi = dist(sagkas[3] , solkas[0])
	diskasAraligi = dist(sagkas[0] , solkas[3])
	kasegimi = abs((sagkas[0][1]-solkas[4][1])/(sagkas[0][0]-solkas[4][0]))

	dudakyuksekligi = dist(agiz[3],agiz[9])
	dudakGenisligi = dist(agiz[0],agiz[6])

	burunGenisligi=dist(burun[4],burun[8])

	gozAcikligiSag = dist(saggoz[1],saggoz[5])
	gozAcikligiSol = dist(solgoz[1], solgoz[5])
	gozAcikligi=(gozAcikligiSag+gozAcikligiSol)/2

	feature=list()

	feature.append(ickasAragligi)
	feature.append(diskasAraligi)
	feature.append(kasegimi)
	feature.append(dudakyuksekligi)
	feature.append(dudakGenisligi)
	feature.append(burunGenisligi)
	feature.append(gozAcikligi)

	return feature

def emotionDetector(img,inc):
	detector = dlib.get_frontal_face_detector()

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(img)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	faces=list()


	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		feature = makAttr(shape)
		faces.append(feature)

	return faces

def gestures(img):
	detector = dlib.get_frontal_face_detector()

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(img)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		cene=shape[:17]
		sagkas=shape[17:22]
		solkas=shape[22:27]
		burun=shape[27:36]
		saggoz=shape[36:42]
		solgoz=shape[42:48]
		agiz=shape[48:67]

		# loop over the face parts individually
		print(face_utils.FACIAL_LANDMARKS_IDXS)
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			# clone the original image so we can draw on it, then
			# display the name of the face part on the image
			clone = image.copy()
			cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)

			# loop over the subset of facial landmarks, drawing the
			# specific face part
			print(name)
			print(i,j)
			print(shape)

			print("yunus")
			for (x, y) in shape[i:j]:
				print((x, y))

				cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

			# extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			# show the particular face part
			cv2.imshow("Image", clone)
			cv2.waitKey(0)

		# visualize all facial landmarks with a transparent overlay
		output = face_utils.visualize_facial_landmarks(image, shape)
		cv2.imshow("Image", output)
		cv2.waitKey(0)


def showPlot(data,path):
	values = np.asarray(data)

	ickasAragligi = values[0:, 0, 0]
	diskasAraligi = values[0:, 0, 1]
	kasegimi = values[0:, 0, 2]

	dudakyuksekligi = values[0:, 0, 3]
	dudakGenisligi = values[0:, 0, 4]

	burunGenisligi = values[0:, 0, 5]

	gozAcikligi = values[0:, 0, 6]

	analyze = list()
	analyze.append(float(np.mean(ickasAragligi)))
	analyze.append(float(np.mean(diskasAraligi)))
	analyze.append(float(np.mean(kasegimi)))
	analyze.append(float(np.mean(dudakyuksekligi)))
	analyze.append(float(np.mean(dudakGenisligi)))
	analyze.append(float(np.mean(burunGenisligi)))
	analyze.append(float(np.mean(gozAcikligi)))


	df = pd.DataFrame({'x': range(0, len(values)),
					   'ickasAragligi': ickasAragligi,
					   'diskasAraligi': diskasAraligi,
					   'kasegimi': kasegimi,
					   'dudakyuksekligi': dudakyuksekligi,
					   'dudakGenisligi': dudakGenisligi,
					   'burunGenisligi': burunGenisligi,
					   'gozAcikligi': gozAcikligi, })

	plt.plot('x', 'ickasAragligi', data=df, marker='o', markerfacecolor=[1, 0, 0], markersize=6, color=[1, 0, 0],
			 linewidth=2)
	plt.plot('x', 'diskasAraligi', data=df, marker='o', markerfacecolor=[0, 1, 0], markersize=6, color=[0, 1, 0],
			 linewidth=2)
	plt.plot('x', 'kasegimi', data=df, marker='o', markerfacecolor=[0, 0, 1], markersize=6, color=[0, 0, 1],
			 linewidth=2)
	plt.plot('x', 'dudakyuksekligi', data=df, marker='o', markerfacecolor=[1, 1, 0], markersize=6, color=[1, 1, 0],
			 linewidth=2)
	plt.plot('x', 'dudakGenisligi', data=df, marker='o', markerfacecolor=[1, 0, 1], markersize=6, color=[1, 0, 1],
			 linewidth=2)
	plt.plot('x', 'burunGenisligi', data=df, marker='o', markerfacecolor=[0, 1, 1], markersize=6, color=[0, 1, 1],
			 linewidth=2)
	plt.plot('x', 'gozAcikligi', data=df, marker='o', markerfacecolor=[1, 1, 1], markersize=6, color=[0, 0, 0],
			 linewidth=2)


	print("ickasAragligi :", round(float(np.mean(ickasAragligi)),2), "\ndiskasAraligi:",
		  round(float(np.mean(diskasAraligi)),2),"\nkasegimi:" , round(float(np.mean(kasegimi)),2),"\ndudakyuksekligi:",
		  round(float(np.mean(dudakyuksekligi)),2),"\ndudakGenisligi:",round(float(np.mean(dudakGenisligi)),2),
		  "\nburunGenisligi:", round(float(np.mean(burunGenisligi)),2),"\ngozAcikligi:",round(float(np.mean(gozAcikligi)),2))
	plt.title(path)
	plt.legend()
	plt.show()

	return analyze


def analysis(path):
	print("-------------------------Begining Analyze of ",path,"-------------------------")
	data = list()
	inc = 1
	for file in glob.glob(path):
		feat = emotionDetector(file, inc)
		inc = inc + 1
		if (len(feat) > 1):
			for i in range(len(feat)):
				data.append(feat[i])
		else:
			data.append(feat)

	return data


def findTres():
	print("-------------------------Begin Of Process-------------------------")
	path = "dataset/happy/*.*"
	data = analysis(path)
	happy=showPlot(data,path)

	print("\n")
	print("-------------------------Begin Of Process-------------------------")
	path = "dataset/sad/*.*"
	data = analysis(path)
	sad=showPlot(data,path)

	print("\n")
	print("-------------------------Begin Of Process-------------------------")
	path = "dataset/disguisting/*.*"
	data = analysis(path)
	disguisting=showPlot(data,path)

	print("\n")
	print("-------------------------Begin Of Process-------------------------")
	path = "dataset/noEmo/*.*"
	data = analysis(path)
	noEmo=showPlot(data,path)

	print("\n")
	print("-------------------------Begin Of Process-------------------------")
	path = "dataset/surprise/*.*"
	data = analysis(path)
	surprise=showPlot(data,path)

	sum1 = happy[0] + sad[0] + disguisting[0] + noEmo[0] + surprise[0]
	sum2 = happy[1] + sad[1] + disguisting[1] + noEmo[1] + surprise[1]
	sum3 = happy[2] + sad[2] + disguisting[2] + noEmo[2] + surprise[2]
	sum4 = happy[3] + sad[3] + disguisting[3] + noEmo[3] + surprise[3]
	sum5 = happy[4] + sad[4] + disguisting[4] + noEmo[4] + surprise[4]
	sum6 = happy[5] + sad[5] + disguisting[5] + noEmo[5] + surprise[5]
	sum7 = happy[6] + sad[6] + disguisting[6] + noEmo[6] + surprise[6]

	happy_tres=np.zeros((1,7))

	happy_tres[0][0] = happy[0] / sum1
	happy_tres[0][1] = happy[1] / sum2
	happy_tres[0][2] = happy[2] / sum3
	happy_tres[0][3] = happy[3] / sum4
	happy_tres[0][4] = happy[4] / sum5
	happy_tres[0][5] = happy[5] / sum6
	happy_tres[0][6] = happy[6] / sum7

	sad_tres = np.zeros((1,7))

	sad_tres[0][0] = sad[0] / sum1
	sad_tres[0][1] = sad[1] / sum2
	sad_tres[0][2] = sad[2] / sum3
	sad_tres[0][3] = sad[3] / sum4
	sad_tres[0][4] = sad[4] / sum5
	sad_tres[0][5] = sad[5] / sum6
	sad_tres[0][6] = sad[6] / sum7

	disguisting_tres = np.zeros((1,7))

	disguisting_tres[0][0] = disguisting[0] / sum1
	disguisting_tres[0][1] = disguisting[1] / sum2
	disguisting_tres[0][2] = disguisting[2] / sum3
	disguisting_tres[0][3] = disguisting[3] / sum4
	disguisting_tres[0][4] = disguisting[4] / sum5
	disguisting_tres[0][5] = disguisting[5] / sum6
	disguisting_tres[0][6] = disguisting[6] / sum7

	noEmo_tres = np.zeros((1,7))

	noEmo_tres[0][0] = noEmo[0] / sum1
	noEmo_tres[0][1] = noEmo[1] / sum2
	noEmo_tres[0][2] = noEmo[2] / sum3
	noEmo_tres[0][3] = noEmo[3] / sum4
	noEmo_tres[0][4] = noEmo[4] / sum5
	noEmo_tres[0][5] = noEmo[5] / sum6
	noEmo_tres[0][6] = noEmo[6] / sum7

	surprise_tres = np.zeros((1,7))

	surprise_tres[0][0] = surprise[0] / sum1
	surprise_tres[0][1] = surprise[1] / sum2
	surprise_tres[0][2] = surprise[2] / sum3
	surprise_tres[0][3] = surprise[3] / sum4
	surprise_tres[0][4] = surprise[4] / sum5
	surprise_tres[0][5] = surprise[5] / sum6
	surprise_tres[0][6] = surprise[6] / sum7

	return happy_tres,sad_tres,disguisting_tres,noEmo_tres,surprise_tres

def findRate(data,tres):
	var1 = data[0]*tres[0][0]
	var1 += data[1] * tres[0][1]
	var1 += data[2] * tres[0][2]
	var1 += data[3] * tres[0][3]
	var1 += data[4] * tres[0][4]
	var1 += data[5] * tres[0][5]
	var1 += data[6] * tres[0][6]

	return var1

def main():
	print("-------------------------Program Begining---------------------------------")
	happy_tres, sad_tres, disguisting_tres, noEmo_tres, surprise_tres=findTres()
	print(happy_tres, sad_tres, disguisting_tres, noEmo_tres, surprise_tres)

	path = "dataset/happy/*.*"
	data = analysis(path)

	happy_rate = findRate(data[0][0],happy_tres)
	sad_rate =  findRate(data[0][0],sad_tres)
	disguisting_rate = findRate(data[0][0], disguisting_tres)
	noEmo_rate = findRate(data[0][0], noEmo_tres)
	surprise_rate = findRate(data[0][0], surprise_tres)

	total_rate=happy_rate+sad_rate+disguisting_rate+noEmo_rate+surprise_rate


	print("happy_rate")
	print(100*((happy_rate)/total_rate))
	print(100*((sad_rate)/total_rate))
	print(100*((disguisting_rate)/total_rate))
	print(100*((noEmo_rate)/total_rate))
	print(100*((surprise_rate)/total_rate))

main()