#!/usr/bin/env python3

# OpenCV program to detect face in real time from webcam footage.
import cv2
from PIL import Image

def make_tflite_face_getter():
    from edgetpu.detection.engine import DetectionEngine
    camera = cv2.VideoCapture(0)
    cameraIndex = 0
    if not camera.isOpened():
        camera = cv2.VideoCapture(1)
        cameraIndex = 1
    
    width, height = int(camera.get(3)), int(camera.get(4))
    
    engine = DetectionEngine('./models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite')
    
    def zoomer():
        nonlocal cameraIndex
        nonlocal camera
        
        if not camera.isOpened():
            camera.release()
            print("[error] couldn't open camera. Aborting and trying new index.")
            cameraIndex += 1
            cameraIndex = cameraIndex % 2
            camera = cv2.VideoCapture(cameraIndex)
            return []
        
        success, img = camera.read()
        if not success:
            print("[error] Could't read from webcam.")
            return []
        
        ans = engine.detect_with_image(
            Image.fromarray(img),
            threshold=0.05,
            keep_aspect_ratio=False,
            relative_coord=False,
            top_k=10
        )
        
        def result_getter(face):
            x, y, x2, y2 = list(map(int, face.bounding_box.flatten().tolist()))
            w = x2 - x
            h = y2 - y
            return (img[y:y + h, x:x + w], (x, y))
        
        if ans:
            return list(map(result_getter, ans))
        return []
    return zoomer, width, height
                
def make_webcam_face_getter():
    # imageCount = 0
    # Hooks up camera to the default video capture device.
    camera = cv2.VideoCapture(0)
    cameraIndex = 0
    if not camera.isOpened():
        camera = cv2.VideoCapture(1)
        cameraIndex = 1

    width, height = int(camera.get(3)), int(camera.get(4))

    # The classifier we use. HAAR is slower than some other options, but
    # is more accurate. We can tune this later.
    faceCascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")

    def zoomer():
        nonlocal cameraIndex
        # nonlocal imageCount
        nonlocal camera

        if not camera.isOpened():
            camera.release()
            print("[error] couldn't open camera. Aborting and trying new index.")
            cameraIndex += 1
            cameraIndex = cameraIndex % 2
            camera = cv2.VideoCapture(cameraIndex)
            return []

        success, img = camera.read()

        if not success:
            print("[error] Could't read from webcam.")
            return []

        greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Perform the detection with some standard params.
        faces = faceCascade.detectMultiScale(
            greyscale,
            minSize=(100, 100),
        )
        if len(faces) == 0:
            return []

        # faces = [max(faces, key=lambda f: f[2] * f[3])] # PICK BIGGEST FACE
        extract_face = lambda f: (img[f[1]:f[1] + f[3], f[0]:f[0] + f[2]], (f[0], f[1]))

        # face_filter = lambda f: f[2] > 100 and f[3] > 100
        # faces = filter(face_filter, faces)

        face_imgs = map(extract_face, faces)

        return list(face_imgs)
        # success = cv2.imwrite("images/" + str(imageCount) + '.jpg', colorFace)

        # if not success:
            # print("[error] Failed to write image to file.")
            # return False
        # imageCount += 1

    return zoomer, width, height
