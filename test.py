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

writer_pipeline_str = "appsrc ! video/x-raw,format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv !\
     video/x-raw(memory:NVMM),format=NV12,width=640,height=640,framerate=24/1 ! nvv4l2h264enc insert-sps-pps=1  \
        insert-vui=1 idrinterval=30 bitrate=1000000 EnableTwopassCBR=1  ! h264parse ! rtph264pay ! udpsink host=192.168.1.34 port=5004 auto-multicast=0"

print(writer_pipeline_str)