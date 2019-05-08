#! /usr/bin/env python
"""
Runner class to see results for input example
"""

import sys
import getopt
import cv2
import dlib

EXPECTED_NUM_IN = 1

frontal_face_detector = dlib.get_frontal_face_detector()


# convenience function from imutils
def dlib_to_cv_bounding_box(box):
    # convert dlib bounding box for OpenCV display
    x = box.left()
    y = box.top()
    w = box.right() - x
    h = box.bottom() - y

    return x, y, w, h


def detect_faces(img):
    # second argument of 1 indicates the image will be upscaled once, upscaling creates a bigger image so it is easier
    # to detect the faces, can increase this number if there are troubles detecting faces
    # returns a bounding box around each face
    detected_faces = frontal_face_detector(img, 1)

    for face in detected_faces:
        # draw box for face
        x, y, w, h = dlib_to_cv_bounding_box(face)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the results
    cv2.imshow("Detected faces: ", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exit_error():
    print('Error: unexpected arguments')
    print('face_detector.py -i <path/to/image>')
    sys.exit()


def main(argv):
    input_img = None

    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        exit_error()

    # only want one input
    if len(args) != EXPECTED_NUM_IN:
        exit_error()

    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            input_img = cv2.imread(arg)
        else:
            exit_error()

    if input_img is None:
        exit_error()

    detect_faces(input_img)


if __name__ == "__main__":
    main(sys.argv[1:])
