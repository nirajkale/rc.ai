import time
import Adafruit_SSD1306   # This is the driver chip for the Adafruit PiOLED
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


class DisplayManager:

    def __init__(self, line_height=10) -> None:
        self.line_height = line_height
        self.disp = Adafruit_SSD1306.SSD1306_128_32(rst=None, i2c_bus=0, gpio=1)
        self.disp.begin()
        self.disp.clear()
        self.disp.display()
        self.width, self.height = self.disp.width, self.disp.height
        self.image = Image.new('1', (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image)
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        self.font = ImageFont.load_default()

    def clear(self):
       self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
       self.disp.image(self.image)
       self.disp.display()

    def draw_text(self, text, x, y, clear=False):
        if clear:
            self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        self.draw.text((x, y), text, font=self.font, fill=255)
        self.disp.image(self.image)
        self.disp.display()

    def print_line(self, text, line_num, left_padding=0, clear=False):
        y = -2 + self.line_height * line_num
        self.draw_text(text, left_padding, y=y, clear=clear)

    def print_lines(self, texts, left_padding=0):
        for line_num, text in enumerate(texts):
            y = -2 + self.line_height * line_num
            self.draw_text(text, left_padding, y=y, clear= line_num==0)


    