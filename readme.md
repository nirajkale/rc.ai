## disable wifi power saving

https://github.com/robwaat/Tutorial/blob/master/Jetson%20Disable%20Wifi%20Power%20Management.md

## disable etrm

https://www.roboticsbuildlog.com/hardware/xbox-one-controller-with-nvidia-jetson-nano

## to install pygame:

#install dependancies
sudo apt-get install python3-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libsdl2-dev libsmpeg-dev python3-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev

#install pygame

python3 -m pip install pygame==2.0.0

## servo config

# to install servoKit
pip3 install adafruit-circuitpython-servokit==1.3.8

detect i2c device: i2cdetect -y -r 1

https://www.jetsonhacks.com/2019/07/22/jetson-nano-using-i2c/


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