import RPi.GPIO as GPIO
from time import sleep


def to_bin(n): return [GPIO.HIGH if ch == '1' else GPIO.LOW for ch in bin(
    n).replace("0b", "").zfill(3)]


GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BOARD)
GPIO.setmode(GPIO.TEGRA_SOC)

# pwm_pin = 32
# s0_pin, s1_pin, s2_pin = 29, 31, 35
pwm_pin1 = 'LCD_BL_PW'
pwm_pin2 = 'GPIO_PE6'
s0_pin, s1_pin, s2_pin = 'SPI2_CS1', 'SPI2_CS0', 'SPI2_MISO'

GPIO.setup([pwm_pin1, pwm_pin2,  s0_pin, s1_pin, s2_pin], GPIO.OUT)
pi_pwm1 = GPIO.PWM(pwm_pin1, 100)  # create PWM instance with frequency
pi_pwm2 = GPIO.PWM(pwm_pin2, 100)

pi_pwm1.start(50)  # start PWM of required Duty Cycle
pi_pwm2.start(50)

input('press to close..')
GPIO.output([s0_pin, s1_pin, s2_pin], (0, 0, 0))
while True:
    str_input = input('>>').strip()
    if str_input == 'exit':
        break
    pin_out = [int(ch) for ch in str_input]
    if len(pin_out) == 3:
        print('setting:', pin_out)
    GPIO.output([s0_pin, s1_pin, s2_pin], pin_out)

GPIO.cleanup()
