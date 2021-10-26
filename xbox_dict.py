from rc_func import INITIAL_ANGLE

SERVO_PITCH = 17
SERVO_YAW = 12
MIN_SERVO_ANGLE = 0
MAX_SERVO_ANGLE = 180
INITIAL_ANGLE = 90
MAX_SERVO_SPEED = 7
MIN_SERVO_SPEED = 0

xbox = {
    "axes":
    {
        3:{
            'angle': INITIAL_ANGLE,
            'usage': 'camera',
            'servo_axis': 'pitch',
            'control': 'left joystick | up-down',
            'joystick_value':0.0,
            'enabled': True,
        },
        2:{
            'angle': INITIAL_ANGLE,
            'usage': 'camera',
            'servo_axis': 'yaw',
            'control': 'left joystick | right-left',
            'joystick_value':0.0,
            'enabled': True,
        },
        1:{
            'angle': INITIAL_ANGLE,
            'usage': 'motion',
            'servo_axis': 'pitch',
            'control': 'right joystick | right-left',
            'joystick_value':0.0,
            'enabled': False,
        },
        0:{
            'angle': INITIAL_ANGLE,
            'usage': 'motion',
            'servo_axis': 'yaw',
            'control': 'right joystick | right-left',
            'joystick_value':0.0,
            'enabled': False,
        },
        4:{
            'control': 'left trigger',
            'enabled': False,
        },
        5:{
            'control': 'right trigger',
            'enabled': False,
        }
    },
    "exit_btn": 11
}