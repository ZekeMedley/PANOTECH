#!/usr/bin/env python3

from webcam_face_extractor import make_webcam_face_getter
from static_generator import make_static_generator

import cv2
import numpy as np
import re
from sys import platform
import subprocess
import random

WINDOW_NAME = 'window'

WIDTH = 768
HEIGHT = 1368

# https://stackoverflow.com/a/1969274
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return int(rightMin + (valueScaled * rightSpan))

def main():
    get_faces, camw, camh = make_webcam_face_getter()
    static_frame = make_static_generator(90, WIDTH, HEIGHT) # should be 1366 for rpi but 1368 is multiple of 4 so...

    # successive_failure_count = 0
    # failure_threshold = 10

    # while True:
    #     success = face_image_writer()
    #     if not success:
    #         successive_failure_count += 1
    #         if successive_failure_count > failure_threshold:
    #             print("[Fatal Error] too many errors in a row!")
    #             # tail call main in desperate effort to maybe fix
    #             main()
    #             break
    #     successive_failure_count = 0

    ### GET SCREEN SIZE??
    ## OSX
    # if platform == 'darwin':
    #     cmd = ['system_profiler', 'SPDisplaysDataType']
    #     cmd_grep = ['grep', 'Resolution']
    #     p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    #     grep = subprocess.Popen(cmd_grep, stdin=p.stdout, stdout=subprocess.PIPE)
    #     p.stdout.close()

    #     res, _ = grep.communicate()
    #     w, h = map(lambda x: int(x), re.findall(r'\d+', res.decode('utf8')))

    # cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    background_red_img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    background_red_img[:] = (12, 12, 165)

    while True:
        faces = get_faces()
        if len(faces):
            aggregate = background_red_img.copy()
            for (f, c) in faces:
                x, y = c
                h, w, _ = f.shape

                x = translate(x, 0, camw - w, 0, WIDTH - w)
                y = translate(y, 0, camh - h, 0, HEIGHT - h)

                random_text = str(random.choice(np.arange(1, 999999)))
                flipped_f = cv2.flip(f, 1)
                f_txtimg = cv2.putText(flipped_f, random_text, (50, 50), cv2.FONT_HERSHEY_TRIPLEX , 1, (0, 255, 0), 2, cv2.LINE_AA)

                aggregate[y: y + h, x: x + w] = np.fliplr(f_txtimg)
            aggregate = np.fliplr(aggregate)
        else:
            # show static
            aggregate = next(static_frame)

        cv2.imshow(WINDOW_NAME, aggregate)
        cv2.waitKey(1)

    # cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
