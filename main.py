#!/usr/bin/env python3

from webcam_face_extractor import make_webcam_face_getter
from static_generator import make_static_generator

import cv2
import numpy as np
import re
from sys import platform
import subprocess

WINDOW_NAME = 'window'

def main():
    get_faces, camw, camh = make_webcam_face_getter()
    static_frame = make_static_generator(100, camw, camh)

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

    background_red_img = np.zeros((camh, camw, 3), np.uint8)
    background_red_img[:] = (12, 12, 165)

    while True:
        faces = get_faces()
        if len(faces):
            aggregate = background_red_img.copy()
            for (f, c) in faces:
                # aggregate = cv2.copyTo(aggregate, f.submat(0, 30, 0, 30))
                x, y = c
                h, w, _ = f.shape

                # if x < 0 or x + w > camw:
                #     continue
                aggregate[y: y + h, x: x + w] = f
            aggregate = np.fliplr(aggregate)
        else:
            # show static
            aggregate = next(static_frame)

        cv2.imshow(WINDOW_NAME, aggregate)
        cv2.waitKey(1)

    # cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
