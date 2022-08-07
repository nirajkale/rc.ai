import cv2
from gst_utils import reader_pipeline, writer_pipeline
from detection_engine_trt import ObjectDetector
from PIL import Image
from display_utils import DisplayManager
import time

FRAME_RATE, WIDTH, HEIGHT = 8, 640, 640
FLAG_OUT_PIPELINE = False

if __name__ == '__main__':

    engine_path = r"/home/niraj/projects/rc.ai/models/best.engine"
    disp = DisplayManager(line_height=12)
    disp.print_line("Loading model ...", line_num=0, clear=True)
    detector = ObjectDetector(engine_path, half=True, labels=['face', 'person'])
    disp.print_lines(["Model loaded", "Starting Gstreamer .."])

    reader_pipeline_str = reader_pipeline(flip_method=0, \
        capture_width= WIDTH, capture_height= HEIGHT,\
        display_width= WIDTH, display_height= HEIGHT, \
        framerate=FRAME_RATE
    )
    writer_pipeline_str = writer_pipeline(host_ip_addr='192.168.1.34', width=WIDTH, height=HEIGHT, port="5004", framerate=FRAME_RATE)
    # print('OUTPUT PIPELINE:', writer_pipeline_str)

    if FLAG_OUT_PIPELINE:
        out = cv2.VideoWriter(writer_pipeline_str, cv2.CAP_GSTREAMER, 0, float(FRAME_RATE), (WIDTH, HEIGHT), True)
    cap = cv2.VideoCapture(reader_pipeline_str, cv2.CAP_GSTREAMER)
    stream_status = "On" if FLAG_OUT_PIPELINE else "Off"
    disp.print_lines([ f"Frame Rate    : {FRAME_RATE}", f"Output Stream : {stream_status}"])

    prev_frame_time = 0
    new_frame_time = 0
    if cap.isOpened():
        while True:
            ret_val, img0 = cap.read()
            if ret_val:
                # print(img.shape)
                new_frame_time = time.time()
                fps = int(1/(new_frame_time-prev_frame_time))
                img1, detections = detector.predict_csi(img0)
                if FLAG_OUT_PIPELINE:
                    out.write(img1)
                print('Detections: %d | FPS: %d\r'%(len(detections), fps), end="")
                prev_frame_time = new_frame_time
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
    else:
        print("Unable to open camera")
    disp.print_line("Bye bye!", line_num=0, clear=True)
