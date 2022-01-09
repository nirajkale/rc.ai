import pygame
import os
from controller import XboxController, JoystickAxisControl, JoystickAccumulator
import Jetson.GPIO as GPIO
from adafruit_servokit import ServoKit

clock = pygame.time.Clock()
joysticks = []


def joystick_init():
    for i in range(0, pygame.joystick.get_count()):
        joysticks.append(pygame.joystick.Joystick(i))
        joysticks[-1].init()
        print("Detected joystick ", joysticks[-1].get_name())
    if len(joysticks) == 0:
        raise Exception('Unable to find joystick')


def configure_pwm_pair(servo_kit: ServoKit, pin_on: int, pin_off, pwm_amgle):
    servo_kit.servo[pin_on].angle = pwm_amgle
    servo_kit.servo[pin_off].angle = 0


def to_bin(n): return [GPIO.HIGH if ch == '1' else GPIO.LOW for ch in bin(
    n).replace("0b", "").zfill(3)]


if __name__ == "__main__":

    pygame.init()
    joystick_init()
    servo_kit = ServoKit(channels=16)
    servo_kit.servo[1].angle = 0
    # mux setup
    pwm_pin1, pwm_pin2 = 'GPIO_PE6', 'LCD_BL_PW'
    s0_pin, s1_pin, s2_pin = 'SPI2_CS1', 'SPI2_CS0', 'SPI2_MISO'
    GPIO.setup([pwm_pin1, pwm_pin2, s0_pin, s1_pin, s2_pin], GPIO.OUT)
    pi_pwm1 = GPIO.PWM(pwm_pin1, 100)
    pi_pwm2 = GPIO.PWM(pwm_pin2, 100)
    pi_pwm1.start(0)
    pi_pwm2.start(0)

    controller = XboxController()
    print('hit RB to exit..')
    camera_yaw = JoystickAccumulator(name='RIGHT_X',
                                     transform_to_int=True,
                                     min_val=-10, max_val=10,
                                     smoothing_factor=0.2,
                                     max_acc_val=180,
                                     min_acc_val=0,
                                     initial_value=90)
    camera_pitch = JoystickAccumulator(name='RIGHT_Y',
                                       transform_to_int=True,
                                       min_val=-10, max_val=10,
                                       smoothing_factor=0.3,
                                       max_acc_val=180,
                                       min_acc_val=0,
                                       initial_value=75)
    left_y = JoystickAxisControl(
        name='LEFT_Y', transform_to_int=True, min_val=-100, max_val=100)
    left_x = JoystickAxisControl(name='LEFT_X')
    # rt = JoystickAxisControl(name='RT', transform_to_int=True, min_val=0, max_val=180)
    # lt = JoystickAxisControl(name='LT', transform_to_int=True, min_val=0, max_val=180)

    # controller.register_joystick_control_by_name(['LEFT_X', 'LEFT_Y', 'RIGHT_X', 'RIGHT_Y'])
    controller.register_joystick_control(camera_pitch)
    controller.register_joystick_control(camera_yaw)
    controller.register_joystick_control(left_y)
    controller.register_joystick_control(left_x)
    keepPlaying = True
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.display.init()
    print('ready to play!')
    while keepPlaying:
        clock.tick(60)
        events = pygame.event.get()
        controller.scan_controller_state(events, verbose=False)
        if controller.current_button_states['RB']:
            keepPlaying = False
        # camera logic
        servo_kit.servo[0].angle = camera_yaw.accumulated_val
        servo_kit.servo[1].angle = camera_pitch.accumulated_val
        # driving logic
        '''
        overall speed is determined by the left_y & the distribution of the speed is determined between left & right wheel
        is determined by left_x
        '''
        left_inclination = int(left_x.raw_val < 0)
        right_inclination = 1 - left_inclination
        inclination_magnitude = abs(left_x.raw_val)
        speed = abs(left_y.transformed_val)
        # figure out distribution of speed
        left_speed = max(
            speed * (1 - (inclination_magnitude * left_inclination)), 0)
        right_speed = max(
            speed * (1 - (inclination_magnitude * right_inclination)), 0)
        pi_pwm1.ChangeDutyCycle(right_speed)
        pi_pwm2.ChangeDutyCycle(left_speed)
        pin_states = None
        if left_y.transformed_val == 0:
            pin_states = (0, 0, 1)  # disable mux output
        if left_y.transformed_val > 0:  # forward
            GPIO.output((s0_pin, s1_pin, s2_pin), (1, 0, 0))
        else:
            GPIO.output((s0_pin, s1_pin, s2_pin), (0, 1, 0))

    GPIO.cleanup()
    pygame.quit()
    print('BYE BYE !!!!!!!!!!!!!!!!')
