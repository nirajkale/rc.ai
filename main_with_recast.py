import pygame
import os
from controller import XboxController, JoystickAxisControl, JoystickAccumulator
import Jetson.GPIO as GPIO
from adafruit_servokit import ServoKit
from display_utils import DisplayManager
import cv2
from gst_utils import reader_pipeline, writer_pipeline
import time
from multiprocessing import Process, Event
from PIL import Image

FRAME_RATE, WIDTH, HEIGHT = 18, 1000, 900 #12, 640, 640
FLAG_OUT_PIPELINE = True
CAPTURE_DIR = r'/home/niraj/projects/rc.ai/data/20-nov-v1'

clock = pygame.time.Clock()
joysticks = []


def joystick_init():
    pygame.joystick.init()
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


class CameraProcessor(Process):

    def __init__(self, exit_event:Event, capture_event:Event, capture_dir:str, out_pipeline:bool=True):
        super().__init__(name='CameraProcess')
        self.exit_event = exit_event
        self.capture_event = capture_event
        self.out_pipeline = out_pipeline
        self.capture_dir = capture_dir
        if not os.path.exists(self.capture_dir):
            os.makedirs(self.capture_dir)
        self.image_count = len(os.listdir(self.capture_dir))

    def run(self) -> None:
        print('starting camera process')
        disp = DisplayManager(line_height=12)
        disp.print_line("Starting gstreamer", line_num=0, clear=True)
        if self.out_pipeline:
            disp.print_line("with out pipeline", line_num=1, clear=True)
        reader_pipeline_str = reader_pipeline(flip_method=0, \
            capture_width= WIDTH, capture_height= HEIGHT,\
            display_width= WIDTH, display_height= HEIGHT, \
            framerate=FRAME_RATE
        )
        writer_pipeline_str = writer_pipeline(host_ip_addr='192.168.1.39', width=WIDTH, height=HEIGHT, port="5004", framerate=FRAME_RATE)
        # print('OUTPUT PIPELINE:', writer_pipeline_str)
        if self.out_pipeline:
            out = cv2.VideoWriter(writer_pipeline_str, cv2.CAP_GSTREAMER, 0, float(FRAME_RATE), (WIDTH, HEIGHT), True)
        cap = cv2.VideoCapture(reader_pipeline_str, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            disp.print_line(f"Camera Failure", line_num=1, clear=True)
        prev_frame_time = 0
        new_frame_time = 0
        last_captured_time = 0
        last_disp_refresh_time = 0
        while not self.exit_event.is_set():
            ret_val, img = cap.read()
            if ret_val:
                out.write(img)
            new_frame_time = time.time()
            fps = int(1/(new_frame_time-prev_frame_time))
            # disp.print_line(f"FPS: {fps}", line_num=0, clear=True)
            prev_frame_time = new_frame_time
            if self.capture_event.is_set() and time.time() - last_captured_time > 3:
                im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                self.image_count += 1
                im.save(os.path.join(self.capture_dir, f'{self.image_count}.jpg'))
                last_captured_time = time.time()
            if time.time() - last_disp_refresh_time >= 5:
                disp.print_lines([f"FPS: {fps}", f'Image Count: {self.image_count}'])
                last_disp_refresh_time = time.time()
        print('stopping camera process')
        cap.release()

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
    left_y = JoystickAxisControl(name='LEFT_Y', transform_to_int=True, min_val=-100, max_val=100)
    left_x = JoystickAxisControl(name='LEFT_X')
    rt = JoystickAxisControl(name='RT', transform_to_int=True, min_val=100, max_val=0) #100 instead of 180 because we're updating duty cycle not angle
    lt = JoystickAxisControl(name='LT', transform_to_int=True, min_val=100, max_val=0)

    # controller.register_joystick_control_by_name(['LEFT_X', 'LEFT_Y', 'RIGHT_X', 'RIGHT_Y'])
    controller.register_joystick_control(camera_pitch)
    controller.register_joystick_control(camera_yaw)
    controller.register_joystick_control(left_y)
    controller.register_joystick_control(left_x)
    controller.register_joystick_control(rt)
    controller.register_joystick_control(lt)
    keepPlaying = True
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.display.init()
    print('starting camera process')
    exit_event = Event()
    capture_event = Event()
    cam_process = CameraProcessor(exit_event= exit_event, capture_event= capture_event, \
        capture_dir= CAPTURE_DIR,\
        out_pipeline= FLAG_OUT_PIPELINE)
    cam_process.start()
    print('ready to play..')
    flag_capture_clear = False
    flag_reset_state = True
    while keepPlaying:
        clock.tick(60)
        events = pygame.event.get()
        controller.scan_controller_state(events, verbose=False)
        if controller.current_button_states['RB']:
            keepPlaying = False
            exit_event.set()
        if controller.current_button_states['B']:
            capture_event.set()
            flag_capture_clear = True
        elif flag_capture_clear:
            # make sure this event is called once
            flag_capture_clear = False
            capture_event.clear()
        # camera logic
        servo_kit.servo[0].angle = camera_yaw.accumulated_val
        servo_kit.servo[1].angle = camera_pitch.accumulated_val
        # driving logic
        '''
        overall speed is determined by the left_y & the distribution of the speed is determined between left & right wheel
        is determined by left_x
        '''
        speed = max(abs(left_y.transformed_val)-5, 0) # add threshold of 5 to avoid l293d overheating at lower pwm duty-cycle
        if speed != 0:
            # left_y gets higher priority over LT & RT
            left_inclination = int(left_x.raw_val < 0)
            right_inclination = 1 - left_inclination
            inclination_magnitude = abs(left_x.raw_val)
            # figure out distribution of speed
            left_speed = max(
                speed * (1 - (inclination_magnitude * left_inclination)), 0)
            right_speed = max(
                speed * (1 - (inclination_magnitude * right_inclination)), 0)
            pi_pwm1.ChangeDutyCycle(right_speed)
            pi_pwm2.ChangeDutyCycle(left_speed)
            if left_y.transformed_val > 0:  # forward
                GPIO.output((s0_pin, s1_pin, s2_pin), (1, 0, 0))
            else:
                GPIO.output((s0_pin, s1_pin, s2_pin), (0, 1, 0))
            flag_reset_state = True
        elif max(rt.transformed_val-5,0) != 0:
            pi_pwm1.ChangeDutyCycle(rt.transformed_val)
            pi_pwm2.ChangeDutyCycle(rt.transformed_val)
            GPIO.output((s0_pin, s1_pin, s2_pin), (1, 1, 0))
            flag_reset_state = True
        elif max(lt.transformed_val-5,0) != 0:
            GPIO.output((s0_pin, s1_pin, s2_pin), (0, 0, 0))
            pi_pwm1.ChangeDutyCycle(lt.transformed_val)
            pi_pwm2.ChangeDutyCycle(lt.transformed_val)
            flag_reset_state = True
        elif flag_reset_state:
            GPIO.output((s0_pin, s1_pin, s2_pin), (0, 0, 1))
            pi_pwm1.ChangeDutyCycle(0)
            pi_pwm2.ChangeDutyCycle(0)
            flag_reset_state = False

    GPIO.cleanup()
    pygame.quit()
    print('BYE BYE !!!!!!!!!!!!!!!!')
