# Main entry point for controlls.
from pyserial import get_serial, send_command
from webcam_face_extractor import make_webcam_face_getter
from time import sleep

def make_direction_decider(width):
    DISTANCE_THRESHOLD = 0.1* width

    def decider(faces):
        try:
            len(faces)
        except:
            return 's'
        if len(faces) == 0:
            return "s"

        print("found " + str(len(faces)) + " faces")
        # Find the face with the minimum distance from the center.
        # f[0] is its x cord, f[2] is its width.
#        target = max(faces, key=lambda f: abs(f[0] + f[2]/2 -
#                                              width/2))

        target = max(faces, key=lambda f: f[2] * f[3])
        distance = target[0] + target[2] / 2 - width/2

        if abs(distance) > DISTANCE_THRESHOLD:
            if target[0] < width / 4:
                return 'L'
            elif target[0] <= width / 2:
                return 'l'  #smaller
            if target[0] > (3*width / 4):
                return 'R'
            elif target[0] >= width / 2:
                return 'r'  # smaller
            # Now trying a different mode of smaller movement
        return "s"

    return decider

def main():
    face_getter, width, height = make_webcam_face_getter()
    controller = make_direction_decider(width)
    ser = get_serial()

    while 1:
        faces = face_getter()
        command = controller(faces)
        print("[sending] " + command)
        ser = send_command(ser, command)
        # rest..
        sleep(2)
        if ( command == 'L' or command == 'R' ):
            ser = send_command(ser, 'D')
            # After this is sent, we need to give the
            # machine some love and let it rest / do its work.
            sleep(2)
        elif ( command == 'l' or command == 'r' ):
            ser = send_command(ser, 'd') # d1 is for a smaller movement
            # After this is sent, we need to give the
            # machine some love and let it rest / do its work.
            sleep(2)

main()

