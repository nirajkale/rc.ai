import cv2
from gst_utils import reader_pipeline, writer_pipeline, reader_pipeline_RGB, writer_pipeline_RGB
from detection_engine_trt import ObjectDetector
from PIL import Image

FRAME_RATE, WIDTH, HEIGHT = 8, 640, 640

if __name__ == '__main__':

    engine_path = r"/home/niraj/projects/rc.ai/models/best.engine"
    # detector = ObjectDetector(engine_path, half=True, labels=['face', 'person'])

    reader_pipeline_str = reader_pipeline_RGB(flip_method=0, \
        capture_width= WIDTH, capture_height= HEIGHT,\
        display_width= WIDTH, display_height= HEIGHT, \
        framerate=FRAME_RATE
    )
    writer_pipeline_str = writer_pipeline(host_ip_addr='192.168.1.34', width=WIDTH, height=HEIGHT, port="5004", framerate=FRAME_RATE)
    print('OUTPUT PIPELINE:', writer_pipeline_str)

    # out = cv2.VideoWriter(writer_pipeline_str, cv2.CAP_GSTREAMER, 0, float(FRAME_RATE), (WIDTH, HEIGHT), True)
    cap = cv2.VideoCapture(reader_pipeline_str, cv2.CAP_GSTREAMER)

    if cap.isOpened():
        while True:
            ret_val, img = cap.read()
            if ret_val:
                # print(img.shape)
                # img1, detections = detector.predict(img, isBGR=True, scaled_inference= False)
                # out.write(img1)
                image = Image.fromarray(img)
                image.save(r'temp.jpg')
                break
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
    else:
        print("Unable to open camera")
