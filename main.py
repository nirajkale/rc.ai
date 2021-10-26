import pygame
import os
from controller import XboxController, JoystickAxisControl, JoystickAccumulator
import Jetson.GPIO as GPIO
from adafruit_servokit import ServoKit

clock = pygame.time.Clock()
# GPIO.setmode(GPIO.BOARD)

joysticks = []
def joystick_init():
    for i in range(0, pygame.joystick.get_count()):
        joysticks.append(pygame.joystick.Joystick(i))
        joysticks[-1].init()
        print("Detected joystick ",joysticks[-1].get_name())
    if len(joysticks) ==0:
        raise Exception('Unable to find joystick')

if __name__ == "__main__":

    pygame.init()
    joystick_init()
    servo_kit = ServoKit(channels=16)
    # GPIO.setup([11, 12, 15, 16], GPIO.OUT, initial=GPIO.LOW)
    controller = XboxController()
    print('hit RB to exit..')
    camera_yaw = JoystickAccumulator(name='RIGHT_X', \
        transform_to_int=True, \
        min_val=-10, max_val=10, \
        smoothing_factor=0.2,\
        max_acc_val = 180,\
        min_acc_val = 0,\
        initial_value = 90)
    camera_pitch = JoystickAccumulator(name='RIGHT_Y', \
        transform_to_int=True, \
        min_val=-10, max_val=10, \
        smoothing_factor=0.3,\
        max_acc_val = 180,\
        min_acc_val = 0,\
        initial_value = 75)
    # controller.register_joystick_control_by_name(['LEFT_X', 'LEFT_Y', 'RIGHT_X', 'RIGHT_Y'])
    controller.register_joystick_control(camera_pitch)
    controller.register_joystick_control(camera_yaw)
    keepPlaying = True
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.display.init()
    while keepPlaying:
        clock.tick(60)
        events = pygame.event.get()
        controller.scan_controller_state(events, verbose=False)
        if controller.current_button_states['RB']:
            keepPlaying = False
        # if controller.joystick_axes_by_name['LEFT_X'].raw_val>0:
        #     GPIO.output([11, 12], (GPIO.LOW, GPIO.HIGH))
        # else:
        #     GPIO.output([11, 12], (GPIO.LOW, GPIO.LOW))
        servo_kit.servo[0].angle = camera_yaw.accumulated_val
        servo_kit.servo[1].angle = camera_pitch.accumulated_val
    GPIO.cleanup()
    pygame.quit()
    print('BYE BYE !!!!!!!!!!!!!!!!')