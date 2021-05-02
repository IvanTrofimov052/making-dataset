import cv2
for i in range(0, 105):
	img = cv2.imread("dataset/screen" + str(i) +".png")
	print(img)
	crop_img = img[150:800, 0:639]
	cv2.imwrite("dataset/screen" + str(i) +".png")
