import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image,ImageOps, ImageChops
import os, sys
from skimage import color, feature
from scipy.spatial import distance
import argparse as ap

names=dict()
names['Adele']=8
names['Babette']=9
names['Cecile']=10
names['Doerte']=12
names['Elsa']=16
names['Fabala']=17
names['Gesa']=18
names['Helvetia']=22
names['Isabella']=25
names['Janette']=31
names['Kiera']=32
names['Letitia']=44

def WhereIsElsa(pic='sow.jpg',name='Elsa'):
    _name=names[name]
    img2 = cv2.imread(pic, 0)
    source = 'data'
    root = os.getcwd() + '\\'
    read = root + source + '\\'+str(_name)+'\\'
    listing = os.listdir(read)
    sift = cv2.xfeatures2d.SIFT_create()
    kp2, des2 = sift.detectAndCompute(img2, None)
    maxim = 0
    filezz = 0
    MIN_MATCH_COUNT = 10
    for file in listing:

        img1 = cv2.imread(read + file, 0)  # queryImage

        # gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        # gray2=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

        # img1=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
        # img2=cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)


        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        bf = cv2.BFMatcher()

        # Match descriptors.
        matches = flann.knnMatch(des1, des2, k=2)
        # Sort them in the order of their distance.
        good = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            seen = set()
            y = np.ones(len(src_pts))

            for i in range(len(src_pts)):
                if (src_pts[i][0][0], src_pts[i][0][0]) in seen:
                    y[i] = 0
                else:
                    seen.add((src_pts[i][0][0], src_pts[i][0][0]))
            seen = set()

            for i in range(len(dst_pts)):
                if (dst_pts[i][0][0], dst_pts[i][0][0]) in seen:
                    y[i] = 0
                else:
                    seen.add((dst_pts[i][0][0], dst_pts[i][0][0]))

            temp1 = [t for i, t in enumerate(src_pts) if y[i] == 1]
            temp2 = [t for i, t in enumerate(dst_pts) if y[i] == 1]

            temp1 = np.float32(temp1).reshape(-1, 1, 2)
            temp2 = np.float32(temp2).reshape(-1, 1, 2)

            temp3 = [t for i, t in enumerate(good) if y[i] == 1]

            sum1 = 0
            for i in range(len(temp1)):
                sum1 = sum1 + distance.euclidean(temp1[0], temp1[1])

            sum2 = 0
            for i in range(len(temp2)):
                sum2 = sum2 + distance.euclidean(temp2[0], temp2[1])

            if sum1 == 0:
                sum1 = 100000
            if sum2 == 0:
                sum2 = 100000
            M, mask = cv2.findHomography(temp1, temp2, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None

        if ((1 - abs(1 - min(sum1 / sum2, sum2 / sum1))) * mask.sum() > maxim):
            maxim = (1 - abs(1 - min(sum1 / sum2, sum2 / sum1))) * mask.sum()
            filezz = file
    img1 = cv2.imread(read + filezz, 0)  # queryImage
    kp1, des1 = sift.detectAndCompute(img1, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params, )
    # Match descriptors.
    matches = flann.knnMatch(des1, des2, k=2)
    # Sort them in the order of their distance.
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        pos = [t for i, t in enumerate(dst_pts) if matchesMask[i] == 1]

        h, w = img1.shape
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    return(np.mean(pos,0))

if __name__ == "__main__":
    parser=ap.ArgumentParser()
    parser.add_argument('-i',"--image",help="Name of picture",required=True)
    parser.add_argument('-n',"--name",help="Name of cow",required=True)
    args=vars(parser.parse_args())

    image_path=args["image"]
    nameOfCow=args["name"]

    print(WhereIsElsa(image_path,nameOfCow))