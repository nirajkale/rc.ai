import Adafruit_SSD1306
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import time

print("Initialize library")
disp = Adafruit_SSD1306.SSD1306_128_32(rst=None, i2c_bus=0, gpio=1)
disp.begin()
# Clear display.
disp.clear()
disp.display()

#sure to create image with mode '1' for 1-bit color.
width = disp.width
height = disp.height
image = Image.new('1', (width, height))
# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)

# Load default font.
font = ImageFont.load_default()
x = 0
padding = -2
top = padding
bottom = height-padding

while True:
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top+8), f"H E L L O  W O R L D !!",  font=font, fill=255)
    
    disp.image(image)
    disp.display()
    time.sleep(1)

print('done')
