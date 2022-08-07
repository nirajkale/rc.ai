import cv2
from datetime import datetime

FRAME_RATE, WIDTH, HEIGHT = 24, 640, 640
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

def reader_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=24,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

reader_pipeline_str = reader_pipeline(flip_method=0, \
    capture_width= 640, capture_height= 640,\
    display_width= 640, display_height= 640
    )

writer_pipeline_str = "appsrc ! video/x-raw,format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv !\
     video/x-raw(memory:NVMM),format=NV12,width=640,height=640,framerate=24/1 ! nvv4l2h264enc insert-sps-pps=1  \
        insert-vui=1 idrinterval=30 bitrate=1000000 EnableTwopassCBR=1  ! h264parse ! rtph264pay ! udpsink host=192.168.1.34 port=5004 auto-multicast=0"

out = cv2.VideoWriter(writer_pipeline_str, cv2.CAP_GSTREAMER, 0, float(FRAME_RATE), (WIDTH, HEIGHT), True)
cap = cv2.VideoCapture(reader_pipeline_str, cv2.CAP_GSTREAMER)

if cap.isOpened():
    while True:
        ret_val, img = cap.read()
        if ret_val:
            # print(img.shape)
            timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            img = cv2.putText(img, timestamp, org, font, fontScale, color, thickness, cv2.LINE_AA)
            out.write(img)
        keyCode = cv2.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 27:
            break
    cap.release()
else:
    print("Unable to open camera")