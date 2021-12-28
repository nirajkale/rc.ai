## disable etrm

https://www.roboticsbuildlog.com/hardware/xbox-one-controller-with-nvidia-jetson-nano

## to install pygame:

#install dependancies
sudo apt-get install python3-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libsdl2-dev libsmpeg-dev python3-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev

#install pygame

python3 -m pip install pygame==2.0.0

## servo config

detect i2c device: i2cdetect -y -r 1

https://www.jetsonhacks.com/2019/07/22/jetson-nano-using-i2c/


## bluetooth config
https://simpleit.rocks/linux/shell/connect-to-bluetooth-from-cli/

