from typing import List
import pygame

BUTTON_ID_NAME_MAPPING = {
    0: 'A', 
    1: 'B',
    3: 'X',
    4: 'Y', 
    6: 'LB', 
    7: 'RB',
    13: 'LJ',
    14: 'RJ'
}

JOY_AXIS_To_NAME_MAPPING = {
    0: 'LEFT_X', 
    1: 'LEFT_Y',
    2: 'RIGHT_X',
    3: 'RIGHT_Y', 
    4: 'RT',
    5: 'LT', 
}

JOY_NAME_TO_ID_MAPPING = {name: id for id,
                          name in JOY_AXIS_To_NAME_MAPPING.items()}


class JoystickAxisControl:

    def __init__(self, name=str, **kwargs) -> None:
        if name not in JOY_NAME_TO_ID_MAPPING:
            raise Exception(
                f"Invalid joy axis name, it should be one of { ''.join(list(JOY_AXIS_To_NAME_MAPPING.values())) }")
        self.name = name
        self.axis = JOY_NAME_TO_ID_MAPPING[name]
        self.transform_to_float = kwargs.get('transform_to_float', False)
        self.transform_to_int = kwargs.get('transform_to_int', False)
        self.max_val = None
        self.min_val = None
        if self.transform_to_float or self.transform_to_int:
            if 'max_val' not in kwargs or 'min_val' not in kwargs:
                raise Exception('transformation max & min value not provided')
            self.max_val = kwargs['max_val']
            self.min_val = kwargs['min_val']
            self.new_range = self.max_val - self.min_val
        self.transformed_val = 0
        self.raw_val = 0.0

    def set_value(self, raw_value):
        self.raw_val = raw_value
        if self.transform_to_float:
            # old range is 1 - (-1) = 2
            self.transformed_val = (
                ((raw_value + 1) * self.new_range) / 2.0) + self.min_val
        elif self.transform_to_int:
            # old range is 1 - (-1) = 2
            self.transformed_val = int(
                (((raw_value + 1) * self.new_range) / 2.0) + self.min_val)

    def post_scan_event(self):
        pass

    def __str__(self) -> str:
        value_to_print = self.raw_val if self.raw_val is not None else 'NaN'
        if (self.transform_to_float or self.transform_to_int) and self.raw_val:
            value_to_print = self.transformed_val
        return f'Joystick {self.name} | axis { self.axis} | value { value_to_print}'


class JoystickAccumulator(JoystickAxisControl):

    def __init__(self, name=str, smoothing_factor=1, **kwargs) -> None:
        self.smoothing_factor = smoothing_factor
        self.initial_value = kwargs.get('initial_value', 90)
        self.max_acc_val = kwargs.get('max_acc_val', 180)
        self.min_acc_val = kwargs.get('min_acc_val', 0)
        self.accumulated_val = self.initial_value
        super().__init__(name=name, **kwargs)

    def update_acc_value(self, change):
        temp = self.accumulated_val + (change * self.smoothing_factor)
        self.accumulated_val = int(
            min(max(temp, self.min_acc_val), self.max_acc_val))

    def set_value(self, raw_value):
        super().set_value(raw_value)
        self.update_acc_value(self.transformed_val)

    def post_scan_event(self):
        self.update_acc_value(self.transformed_val)

    def __str__(self) -> str:
        return f'Joystick {self.name} | transformed_val {self.transformed_val} | accumelated_val {self.accumelated_val}'


class XboxController:

    def __init__(self) -> None:
        self.registered_joystick_axes = {}
        self.joystick_axes_by_name = {}
        self.current_button_states = {
            name: False for name in BUTTON_ID_NAME_MAPPING.values()}
        self.last_button_states = dict(self.current_button_states)
        self.flipped_buttons = dict(self.current_button_states)

    def register_joystick_control(self, joystick_control: JoystickAxisControl):
        self.registered_joystick_axes[joystick_control.axis] = joystick_control
        self.joystick_axes_by_name[joystick_control.name] = joystick_control

    def register_joystick_control_by_name(self, names: List[str]):
        for name in names:
            self.register_joystick_control(JoystickAxisControl(name=name))

    def scan_controller_state(self, events, verbose=False):
        self.last_button_states = dict(self.current_button_states)
        for event in events:
            if event.type == pygame.JOYAXISMOTION:
                if event.axis in self.registered_joystick_axes:
                    control = self.registered_joystick_axes[event.axis]
                    control.set_value(event.value*-1)
                    if verbose:
                        print(control)
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button in BUTTON_ID_NAME_MAPPING:
                    btn_name = BUTTON_ID_NAME_MAPPING[event.button]
                    self.current_button_states[btn_name] = True
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in BUTTON_ID_NAME_MAPPING:
                    btn_name = BUTTON_ID_NAME_MAPPING[event.button]
                    self.current_button_states[btn_name] = False
        for joystick_control in self.registered_joystick_axes.values():
            joystick_control.post_scan_event()
