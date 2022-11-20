import pygame
import os
pygame.init()

clock = pygame.time.Clock()
from xbox_dict import xbox
from rc_func import (
    MIN_SERVO_ANGLE, MAX_SERVO_ANGLE,
    MAX_SERVO_SPEED, MIN_SERVO_SPEED
)

joysticks = []
def joystick_init():
    pygame.joystick.init()
    j_count = 0
    for i in range(0, pygame.joystick.get_count()):
        joysticks.append(pygame.joystick.Joystick(i))
        joysticks[-1].init()
        print("Detected joystick ",joysticks[-1].get_name())
        j_count += 1
    if j_count ==0:
        raise Exception('Unable to find joystick')

def map_joystick_to_servo_speed(value, precision = 2):
    value = round(value, precision)
    if value >0:
        return value *( MAX_SERVO_SPEED - MIN_SERVO_SPEED) + MIN_SERVO_SPEED
    else:
        return value *( MAX_SERVO_SPEED - MIN_SERVO_SPEED) - MIN_SERVO_SPEED

def set_servo_angle(host,):
    pass

def main_app(host = None):
    keepPlaying = True
    last_axis = -1
    # os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.display.init()
    joystick_init()
    while keepPlaying:
        clock.tick(60)
        for event in pygame.event.get():
            #The zero button is the 'a' button, 1 the 'b' button, 3 the 'y' 
            #button, 2 the 'x' button
            if event.type == pygame.JOYAXISMOTION:
                print('axis: ', event.axis, ' | value:', event.value)
            elif event.type == pygame.JOYBUTTONDOWN:
                print('btn down:', event.button)
            elif event.type == pygame.JOYBUTTONUP:
                print('btn up:', event.button)
    pygame.quit()

if __name__ == "__main__":

    host = None
    main_app(host= host)