# %%
import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

# %%
# ۱. ایجاد تصویر متن با Pillow
img = Image.new('RGB', (200,150), color='white')
draw = ImageDraw.Draw(img)
# font = ImageFont.truetype('IranNastaliq.ttf', 32)
font = ImageFont.truetype('Vazirmatn-Regular.ttf', 32)
draw.text((20, 20), "سلام", font=font, fill='black')
draw.text((20, 52), "خدا حافظ", font=font, fill='black')
# %%
# draw.ellipse([(10,5), (12,7)], fill='red')
draw.circle(xy = (20, 20), 
            radius = 1,
            fill = (0, 127, 0),
            )

# %%
# show img with plt
plt.imshow(np.array(img))
plt.axis("off")
plt.show()
# %%

# %%
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

zwj = "\u200d"

steps = [
    "س" + zwj,
    "س" + zwj + "ل" + zwj,
    "س" + zwj + "ل" + zwj + "ا" ,
    "س" + zwj + "ل" + zwj + "ا"  + "م",
]

font = ImageFont.truetype("Vazirmatn-Regular.ttf", 48)

img = Image.new("RGB", (300, 250), "white")
draw = ImageDraw.Draw(img)

y = 10
for s in steps:
    draw.text((20, y), s, font=font, align ="right", fill="black", direction="rtl",  # Pillow ≥10.0
              features=["-liga", "+calt"])  # برای کنترل اتصال در فونت‌هایی که OpenType دارند
    y += 55

plt.imshow(np.array(img)); plt.axis("off"); plt.show()

# %%
import freetype
import arabic_reshaper
from bidi.algorithm import get_display

# ۱. بارگذاری فونت و تنظیم اندازه
import freetype
from arabic_reshaper import ArabicReshaper
from bidi.algorithm import get_display

# ۱. بارگذاری فونت
face = freetype.Face("/usr/share/fonts/truetype/Vazirmatn-Regular.ttf")
face.set_char_size(48 * 64)

# ۲. reshaper با merge_ligatures=False
config    = {'merge_ligatures': False}
reshaper  = ArabicReshaper(config)

raw_text  = "سلام"
reshaped  = reshaper.reshape(raw_text)
bidi_text = get_display(reshaped)

# ۳. محاسبه‌ی bbox برای هر گلیف
x_cursor = 0
y_cursor = face.size.ascender >> 6

bboxes = []
for ch in bidi_text:
    face.load_char(ch)
    g = face.glyph
    x0 = x_cursor + g.bitmap_left
    y0 = y_cursor - g.bitmap_top
    x1 = x0 + g.bitmap.width
    y1 = y0 + g.bitmap.rows
    bboxes.append((ch, (x0, y0, x1, y1)))
    x_cursor += (g.advance.x >> 6)

print(bboxes)
# خروجیِ bboxes باید ۴ تا tuple برای س، ل، ا و م بدهد.


# %%
from PIL import Image, ImageDraw

# ساخت کادر سفید با عرض مجموع آوانس‌ها
total_width = x_cursor
height      = (face.size.ascender + abs(face.size.descender)) >> 6

img = Image.new("RGB", (total_width, height), "white")
draw = ImageDraw.Draw(img)

# رسم یک مستطیل دور هر باکس
for _, (x0,y0,x1,y1) in bboxes:
    draw.rectangle([x0, y0, x1, y1], outline="red")

# رسم متن اصلی در پس‌زمینه تا خود گلیف‌ها هم دیده شوند
draw.text((0,0), bidi_text, font=None, fill="black", anchor="lt")  # font نداریم در PIL

img.show()

# %%
