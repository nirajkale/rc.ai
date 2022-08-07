import cv2
from gst_utils import reader_pipeline
from detection import ObjectDetector

if __name__ == '__main__':

    reader_pipeline_str = reader_pipeline(flip_method=0, \
        capture_width= 640, capture_height= 640,\
        display_width= 640, display_height= 640
    )

    cap = cv2.VideoCapture(reader_pipeline_str, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        while True:
            ret_val, img = cap.read()
            if ret_val:
                # print(img.shape)
                
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
    else:
        print("Unable to open camera")
