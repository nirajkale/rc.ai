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


def writer_pipeline(host_ip_addr:str, width:str, height:str, port:str="5004", framerate:str="24"):
    return f"appsrc ! video/x-raw,format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! \
    video/x-raw(memory:NVMM),format=NV12,width={width},height={height},framerate={framerate}/1 ! nvv4l2h264enc insert-sps-pps=1 \
    insert-vui=1 idrinterval=30 bitrate=1000000 EnableTwopassCBR=1  ! h264parse ! rtph264pay ! udpsink host={host_ip_addr} port={port} auto-multicast=0"


if __name__ == '__main__':

    print(writer_pipeline(host_ip_addr='192.168.1.1'))