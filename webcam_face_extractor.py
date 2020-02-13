# OpenCV program to detect face in real time from webcam footage.
import cv2

def make_webcam_face_getter():
    imageCount = 0
    # Hooks up camera to the default video capture device.
    camera = cv2.VideoCapture(0)
    cameraIndex = 0
    if not camera.isOpened():
        camera = cv2.VideoCapture(1)
        cameraIndex = 1

    # The classifier we use. HAAR is slower than some other options, but
    # is more accurate. We can tune this later.
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def zoomer():
        nonlocal cameraIndex
        nonlocal imageCount
        nonlocal camera

        if not camera.isOpened():
            camera.release()
            print("[error] couldn't open camera. Aborting and trying new index.")
            cameraIndex += 1
            cameraIndex = cameraIndex % 2
            camera = cv2.VideoCapture(cameraIndex)
            return False

        success, img = camera.read()

        if not success:
            print("[error] Could't read from webcam.")
            return False

        greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Perform the detection with some standard params.
        faces = faceCascade.detectMultiScale(greyscale)
        if len(faces) == 0:
            return True

        x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
        colorFace = img[y:y + h, x:x + w]
        success = cv2.imwrite("images/" + str(imageCount) + '.jpg', colorFace)

        if not success:
             print("[error] Failed to write image to file.")
             return False
        imageCount += 1

        return True

    return zoomer
