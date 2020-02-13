import cv2

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("[error] couldn't open camera.")
    exit()
    
frontalCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    success, img = camera.read()
    if not success:
        print("[error] failed to get image from camera.");
        exit();
        
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # These params are made up and tuneable.
    faces = frontalCascade.detectMultiScale(
            greyscale,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('image', img)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break