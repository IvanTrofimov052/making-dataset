from PIL import Image
from matplotlib import pyplot as plt
import cv2.aruco as aruco



class opencv:
    def show_image(image):
        cv2.imshow('image window', image)
        cv2.waitKey(0)


        # cv2.destoyAllWindows()

    def get_image(path):
        image = cv2.imread(path)

        return image

    def save_img(image, path):
        cv2.imwrite(path, image)

    def change_to_hsv(path):
        image = cv2.imread(path)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        return hsv

    def make_mask(path, lower, upper):
        image = opencv.get_image(path)
        hsv = opencv.change_to_hsv(path)

        # define range of blue color in HSV
        lower_blue = np.array(lower)
        upper_blue = np.array(upper)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if(len(contours) >= 1):
            c = sorted(contours, key=cv2.contourArea, reverse=True)

            for cout in c:
                # compute the rotated bounding box of the largest contour
                rect = cv2.minAreaRect(cout)
                box = np.int0(cv2.boxPoints(rect))

                cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

        return image

    def find_cone(path):
        lower = [0, 135, 135]
        upper = [15, 255, 255]

        return opencv.make_mask(path, lower, upper)

    def line_detection_method_1(path):
        image = opencv.get_image(path)

        edges = cv2.Canny(image, 50, 150)
        rho_accuracy = 1
        theta_accuracy = np.pi / 180
        min_length = 200
        lines = cv2.HoughLines(edges, rho_accuracy, theta_accuracy, min_length)

        for line in lines:
            rho, theta = line[0]
            a = np.cos ( theta )
            b = np.sin ( theta )
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return image

    def line_detection_method_2(path):
        lower = [0, 0, 255]
        upper = [255, 255, 255]

        return opencv.make_mask ( path, lower, upper)

    def sift(path, path_1):
        img1 = cv2.imread(path, 0)  # queryImage
        img2 = cv2.imread(path_1, 0)  # trainImage

        img4 = cv2.imread(path_1)

        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None )
        kp2, des2 = sift.detectAndCompute(img2, None )

        # # BFMatcher with default params
        bf = cv2.BFMatcher ()
        matches = bf.knnMatch ( des1, des2, k=2 )

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        list_kp1 = []
        list_kp2 = []

        # For each match...
        for mat in good:
            # Get the matching keypoints for each of the images
            img1_idx = mat[0].queryIdx
            img2_idx = mat[0].trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            # Append to each list
            list_kp1.append ( (x1, y1) )
            list_kp2.append ( (x2, y2) )

        list_kp2 = sorted(list_kp2)

        max_x = list_kp2[0][0]
        max_y = list_kp2[0][0]
        min_x = list_kp2[0][0]
        min_y = list_kp2[0][0]

        for coord in list_kp2:
            if coord[0] > max_x:
                max_x = coord[0]
            elif coord[0] < min_x:
                min_x = coord[0]

            if coord[1] > max_y:
                max_y = coord[1]
            elif coord[1] < min_y:
                min_y = coord[1]

        strat_point = (int(min_x), int(max_y))
        end_point =(int(max_x), int(min_y))

        color = (255, 0, 0)

        img4 = cv2.rectangle(img4, strat_point, end_point,(15,255,255),3)

        draw_params = dict ( matchColor=(0, 255, 0),
                             singlePointColor=(255, 0, 0),
                             matchesMask=good,
                             flags=0 )

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, outImg=None)

        return img4

    def making_masked_image(path, lower, upper):
        image = opencv.get_image ( path )
        hsv = opencv.change_to_hsv ( path )

        # define range of blue color in HSV
        lower_blue = np.array ( lower )
        upper_blue = np.array ( upper )

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange ( hsv, lower_blue, upper_blue )

        # Apply the mask and display the result
        maskedImg = cv2.bitwise_and(image, image, mask=mask)

        cv2.imwrite('1.jpg', maskedImg)

        new_image = opencv.sift(path, '2.png')

        return maskedImg

    def detect_arco_marker(path):
        image = opencv.get_image(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
        arucoParameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=arucoParameters)
        image = aruco.drawDetectedMarkers(image, corners)

        return image
        
    def make_mask_1(path, lower, upper):
        image = opencv.get_image(path)
        hsv = opencv.change_to_hsv(path)

        # define range of blue color in HSV
        lower_blue = np.array(lower)
        upper_blue = np.array(upper)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        image = cv2.bitwise_and(image, image, mask)

        return image
        
for i in range(0, 321):
	image = opencv.make_mask_1("dataset/screen" + str(i) + ".png", [20, 100, 100], [30, 255, 255])
	opencv.save_img(image, "dataset_1/screen" + str(i) + ".png")
