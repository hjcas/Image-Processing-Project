# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import cv2

BOUNDED_BOX_Y_OFFSET = 0

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def arg_parser_init():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
    ap.add_argument("-w", "--width", type=float, required=True,
        help="width of the left-most object in the image (in inches)")

    args = vars(ap.parse_args())

    return args



def get_gray_scale_image(args):
# load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(args["image"])
    image_height, image_width, _ = image.shape
    cropped_image = image[80:int(image_width / 2), 20: int(image_height / 2)]
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    cv2.imshow('greySCALE', gray)
    cv2.waitKey(0)

    return cropped_image, gray



def perform_edge_detection_and_find_contours(gray):
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
# edged = cv2.Canny(gray, 50, 100)
    edged = cv2.Canny(gray, 50, 200)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts


def calculate_ppi(cnts, image, args):
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

# loop over the contours individually
    count = 0
    for c in cnts:
        if (count > 4):
            break

        count += 1

        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue

        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # draw the midpoints on the image
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 2)

            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, inches)
            if pixelsPerMetric is None:
                pixelsPerMetric = dB / args["width"]
            # compute the size of the object
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            # draw the object sizes on the image
            cv2.putText(orig, "{:.1f}in".format(dimA),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            cv2.putText(orig, "{:.1f}in".format(dimB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)

            # show the output image
        cv2.imshow("Image", orig)
        cv2.waitKey(0)
        
    cv2.imshow("Image", orig)
    cv2.waitKey(0)
    return pixelsPerMetric


def calibrate_and_get_ppi(args):
    image, gray_image = get_gray_scale_image(args)
    cnts = perform_edge_detection_and_find_contours(gray_image)
    ppi = calculate_ppi(cnts, image, args)

    return ppi
#============================================================================
#============================================================================
#============================================================================
#Height Measurement==========================================================

def measure_height(args, ppi):
    hog = init_hog_detector()
    image = cv2.imread(args["image"])
    height_inches = process_image(image, args["image"], hog, ppi)
    FormatHeigh(height_inches)

def FormatHeigh(height_inches):
    # height_inches = height_inches + 0.5
    height = int(height_inches + 0.5)

    feet = int(height / 12)
    inches = height % 12

    print(f'We estimate you are {feet}\'{inches} ft.')


def process_image(image, imagePath, hog, ppi):
	image = imutils.resize(image, width=min(800, image.shape[1]))
	orig = image.copy()

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(1, 1), scale=1.05)
    
	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		# cv2.rectangle(image, (xA, yA + 45), (xB, yB - 50), (0, 255, 0), 2)
		cv2.rectangle(image, (xA, yA + BOUNDED_BOX_Y_OFFSET), (xB, yB - BOUNDED_BOX_Y_OFFSET), (0, 255, 0), 2)

	# show some information on the number of bounding boxes
	filename = imagePath[imagePath.rfind("/") + 1:]
	print("[INFO] {}: {} original boxes, {} after suppression".format(
		filename, len(rects), len(pick)))

	# show the output images
	# cv2.imshow("Before NMS", orig)
	cv2.imshow("After NMS", image)
	cv2.waitKey(0)
	return ((yB - yA)/ ppi)

def init_hog_detector():
# initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    return hog




#============================================================================
#============================================================================

def main():
   args = arg_parser_init()
   ppi = calibrate_and_get_ppi(args)

   if ppi is None:
       print(f'ppi: {ppi}')
   else:
       measure_height(args, ppi)


if __name__ == '__main__':
    main()
