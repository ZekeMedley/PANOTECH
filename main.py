#!/usr/bin/env python3

from webcam_face_extractor import make_webcam_face_getter

def main():
    face_image_writer = make_webcam_face_getter()
    
    successive_failure_count = 0
    failure_threshold = 10
    
    while True:
        success = face_image_writer()
        if not success:
            successive_failure_count += 1
            if successive_failure_count > failure_threshold:
                print("[Fatal Error] too many errors in a row!")
                # tail call main in desperate effort to maybe fix
                main()
        successive_failure_count = 0
        
main()
