import cv2
from detection import ObjectDetector
import time

def gstreamer_pipeline(
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

pipeline = gstreamer_pipeline(flip_method=0, \
    capture_width= 640, capture_height= 640,\
    display_width= 640, display_height= 640
    )

model_path = r"models/yolov5s.onnx"
# model_path = r"models/yolov5s_fp16.onnx"
detector = ObjectDetector(model_path)
detector.load_coco_labels()
detector.perform_warmup(steps=5)

prev_frame_time = 0
new_frame_time = 0
print(pipeline)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if cap.isOpened():
    while True:
        ret_val, img = cap.read()
        img, detections = detector.predict(img, isBGR=True, scaled_inference= True)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        new_frame_time = time.time()
        fps = int(1/(new_frame_time-prev_frame_time))
        print('Detections: %d | FPS: %d\r'%(len(detections), fps), end="")
        prev_frame_time = new_frame_time
        # This also acts as
        keyCode = cv2.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 27:
            break
    cap.release()
else:
    print("Unable to open camera")