## disable wifi power saving

https://github.com/robwaat/Tutorial/blob/master/Jetson%20Disable%20Wifi%20Power%20Management.md

## disable bluetooth etrm

https://forums.developer.nvidia.com/t/disabling-ertm-permanently-in-jetpack-4-4-ubuntu-18-04-on-nano-4gb/159567

## to install pygame:

#install dependancies
sudo apt-get install python3-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libsdl2-dev libsmpeg-dev python3-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev

# install pygame

    python3 -m pip install pygame==2.0.0

# to install servoKit

    pip3 install adafruit-circuitpython-servokit==1.3.8 or 

    sudo pip3 install -U \
    adafruit-circuitpython-busdevice==5.1.2 \
    adafruit-circuitpython-motor==3.3.5 \
    adafruit-circuitpython-pca9685==3.4.1 \
    adafruit-circuitpython-register==1.9.8 \
    adafruit-circuitpython-servokit==1.3.8 \
    Adafruit-Blinka==6.11.1 \
    Adafruit-GPIO==1.0.3 \
    Adafruit-MotorHAT==1.4.0 \
    Adafruit-PlatformDetect==3.19.6 \
    Adafruit-PureIO==1.1.9 \
    Adafruit-SSD1306==1.6.2

    detect i2c device: i2cdetect -y -r 1

    https://www.jetsonhacks.com/2019/07/22/jetson-nano-using-i2c/

# enable pwm on nano for motor control

    ** run: sudo /opt/nvidia/jetson-io/jetson-io.py
    ** select: Configure Jetson 40pin Header
    ** then select: Configure header pins manually
    ** enable pwm0 & pwm2 then save & reboot
 
# install Adafruit_SSD1306 for display

    pip3 install Adafruit-SSD1306

## setup xbox controller

https://pimylifeup.com/xbox-controllers-raspberry-pi/



## bluetooth config

    https://simpleit.rocks/linux/shell/connect-to-bluetooth-from-cli/

## stuff related to vision

# install pytorch - needed for moving tensors to CUDA device

    use tutorial https://qengineering.eu/install-pytorch-on-jetson-nano.html
    & install torch 1.9.0 & torchvision 0.10

# set yolov5 

    ** clone yolov5 repo
    git clone https://github.com/ultralytics/yolov5.git
    ** comment out certain libs to avoid conflict: opencv-python>=4.1.1, torch, torchvision, matplotlib, seaborn & pandas
    ** then install requirements.txt 
    pip3 install -r requirements.txt --no-deps --verbose
        *** --no-deps is important to avoid re-installation of torch
    

# install onnx

    ** install libs needed to build onnx wheel
    sudo apt-get install cmake==3.2
        ** or alternaticaly, you can download .sh (>= v3.23) from https://cmake.org/download/ 
        ** this shell file will unzip bin files (which will contain cmake execuatable) for whcih you can add path in ~/.bashrc
        ** contents can be copied using: sudo cp -r cmake-3.24.0-linux-aarch64/ /usr/bin/
        ** uninstall existing installation sudo apt-get remove cmake -y
        ** add path to new installation/copy
            - sudo nano ~/.bashrc
            - add this line PATH=$PATH:/usr/bin/cmake-3.24.0-linux-aarch64/bin
            - apply changes source ~/.bashrc
        
    sudo apt-get install protobuf-compiler
    sudo apt-get install libprotoc-dev

    ** install a specific version of onnx
    pip3 install onnx==1.9.0    

    ** if you get error "protobuf requires Python '>=3.7' but the running Python is 3.6.9", 
    then try upgrading pip: pip3 install --upgrade pip


# export model to tensorrt 

    python3 export.py --weights /home/niraj/projects/rc.ai/models/best.pt --imgsz 640 --batch-size 1 --device 0 --half --simplify --include engine

# gstreamer command for video reception

    gst-launch-1.0 -v udpsrc port=5004 caps = “application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96” ! rtph264depay ! decodebin ! videoconvert ! autovideosink


# start video (without opencv)

    nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)640, format=(string)NV12, framerate=(fraction)12/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)640, height=(int)640, format=(string)BGRx ! videoconvert ! nvv4l2h264enc insert-sps-pps=1 insert-vui=1 idrinterval=30 bitrate=1000000 EnableTwopassCBR=1  ! h264parse ! rtph264pay ! udpsink host=192.168.1.39 port=5004 auto-multicast=0

## quick command to connect xbox controller once it is setup

    sudo bluetoothctl //this should take you to bt terminal

### if xbox max does not appear already the scan for it using:

    agent on
    default-agent

## then connect using 

    connect MAC

## trust

    trust MAC