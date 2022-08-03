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

- use modified version of this tutorial on: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html 

    1) sudo apt-get -y update;
    2) sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev;
    3) export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5+nv22.01-cp36-cp36m-linux_aarch64.whl
    4) python3 -m pip install --upgrade pip; python3 -m pip install expecttest xmlrunner hypothesis aiohttp numpy=='1.19.4' pyyaml scipy=='1.5.3' ninja cython typing_extensions protobuf; export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"; python3 -m pip install --upgrade protobuf; python3 -m pip install --no-cache $TORCH_INSTALL


# install torch-vision

    pip3 install torchvision==0.11.3

