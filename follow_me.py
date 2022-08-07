import cv2
from gst_utils import reader_pipeline, writer_pipeline
from detection_engine_trt import ObjectDetector
from PIL import Image
from display_utils import DisplayManager
import time
import gc
import traceback

FRAME_RATE, WIDTH, HEIGHT = 8, 640, 640
FLAG_OUT_PIPELINE = True

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
    try:
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
                    cx, cy, area = -1, -1, -1
                    det_persons = [det for det in detections if det.label == 'person' and det.prob >= 0.2]
                    if len(det_persons) > 0:
                        person_in_focus = det_persons[0]
                        person_in_focus.normalize_dims(WIDTH, HEIGHT)
                        cx, cy, area = person_in_focus.center_x, person_in_focus.center_y, person_in_focus.normalized_area
                    print('Detections: %d | FPS: %d | Person: %.1f x %.1f x %.1f\r'%(len(detections), fps, cx, cy, area), end="")
                    prev_frame_time = new_frame_time
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
        else:
            print("Unable to open camera")
        disp.print_line("Bye bye!", line_num=0, clear=True)
    except KeyboardInterrupt:
        print('Keyboard interrupt received')
    except Exception as e:
        print('Error in vision loop')
        traceback.print_exc()
    finally:
        del detector
        cap.release()
        out.release()
        gc.collect()