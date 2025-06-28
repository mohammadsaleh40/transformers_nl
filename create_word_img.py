import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

# -------------------------------
# Configuration
# -------------------------------
IMAGE_HEIGHT = 50
IMAGE_WIDTH = 128
NUM_IMAGES = 1_000_000
IMAGES_PER_SHARD = 100_000
NUM_SHARDS = NUM_IMAGES // IMAGES_PER_SHARD
OUTPUT_DIR = "persian_synth_shards"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# حروف فارسی
PERSIAN_ALPHABET = list("ابتثجحخدذرزسشصضطظعغفقکگلمنوهی")

# مسیر فونت‌ها
font_paths = [
    "danstevis.otf",
    "Amiri Quran Colored.ttf",
    "Far.Yas.ttf",
    "B Moj.ttf",
    "A Iranian Sans.ttf",
    "B Koodak.ttf",
    "Mj_Meem Medium.ttf",
    "2 Hamid_YasDL.com.ttf"
]

# (اختیاری) لیست پس‌زمینه‌های واقعی برای ترکیب
# bg_paths = ["bg1.jpg", "bg2.jpg", ...]


# -------------------------------
# ۱. تولید کلمهٔ فارسی تصادفی
# -------------------------------
def random_persian_word(min_len=1, max_len=14):
    L = random.randint(min_len, max_len)
    return "".join(random.choices(PERSIAN_ALPHABET, k=L))


# -------------------------------
# ۲. رندر متن + افکت حاشیه/سایه
# -------------------------------
def render_text_image(text, font_path):
    # بوم RGB سفید
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # اندازهٔ فونت تصادفی
    sz = random.randint(17, 20)
    font = ImageFont.truetype(font_path, sz)
    
    # ابعاد متن
    mask = font.getmask(text)
    w, h = mask.size
    
    # موقعیت در مرکز با جابجایی کوچک
    x = (IMAGE_WIDTH - w) // 2 + random.randint(-5, 5)
    y =  (IMAGE_HEIGHT  - h) // 2 + random.randint(-3, 3)
    
    # رسم حاشیه (outline)
    outline_color = (random.randint(50,200),) * 3
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        draw.text((x+dx, y+dy), text, font=font, fill=outline_color)
    # رسم متن اصلی
    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    
    return img


# -------------------------------
# ۳. تحریف پرسپکتیو و هندسی
# -------------------------------
def apply_perspective(img):
    w, h = img.size
    # چهار نقطهٔ ورودی
    src = [(0,0), (w,0), (w,h), (0,h)]
    # چهار نقطهٔ خروجی با جابجایی تصادفی کوچک
    delta = 0.1 * min(w,h)
    dst = [
        (random.uniform(0,delta), random.uniform(0,delta)),
        (random.uniform(w-delta,w), random.uniform(0,delta)),
        (random.uniform(w-delta,w), random.uniform(h-delta,h)),
        (random.uniform(0,delta), random.uniform(h-delta,h))
    ]
    coeffs = ImageTransform.coeffs(src, dst)
    return img.transform((w,h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

# helper برای محاسبهٔ ماتریس پریسپکتیو
class ImageTransform:
    @staticmethod
    def coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0,0,0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0,0,0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.matrix(matrix, dtype=float)
        B = np.array(pb).reshape(8)
        res = np.linalg.lstsq(A, B, rcond=None)[0]
        return list(res) + [1.0]


# -------------------------------
# ۴. سایر اعوجاج‌ها + بلور
# -------------------------------
def random_distort(img):
    # چرخش
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, fillcolor=(255,255,255))
    # شیب
    w, h = img.size
    m = np.tan(random.uniform(-0.1, 0.1))
    xshift = abs(m)*h
    img = img.transform(
        (w, h),
        Image.AFFINE, (1, m, -xshift if m>0 else 0, 0, 1, 0),
        fillcolor=(255,255,255)
    )
    # بلور
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0,1.0)))
    return img


# -------------------------------
# ۵. ترکیب پس‌زمینه (اختیاری)
# -------------------------------
# def blend_background(text_img):
#     # اگر bg_paths خالی است از همین متن استفاده شود
#     bg = Image.open(random.choice(bg_paths)).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
#     α = random.uniform(0.3,0.7)
#     return Image.blend(bg, text_img, α)



# -------------------------------
# ۶. تبدیل نهایی به Grayscale + نویز
# -------------------------------
def finalize(img):
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)
    # نویز نمکی-فلفلی
    mask_sp = np.random.choice((0,1,2), size=arr.shape, p=[0.98,0.01,0.01])
    arr = np.where(mask_sp==1, 255, arr)
    arr = np.where(mask_sp==2, 0, arr)
    return Image.fromarray(arr)


# -------------------------------
# TFRecord helpers
# -------------------------------
def serialize_example(img_np, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_np.tobytes()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')])),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feature))
    return ex.SerializeToString()


# -------------------------------
# نوشتن شاردهای TFRecord
# -------------------------------
def write_sharded_tfrecords():
    for sid in range(NUM_SHARDS):
        start = sid * IMAGES_PER_SHARD
        end   = start + IMAGES_PER_SHARD
        path  = os.path.join(OUTPUT_DIR, f"p_{sid:03d}-of-{NUM_SHARDS:03d}.tfrecord")
        with tf.io.TFRecordWriter(path) as writer:
            for i in range(start, end):
                word = random_persian_word()
                font = random.choice(font_paths)
                img  = render_text_image(word, font)
                img  = apply_perspective(img)
                img  = random_distort(img)
                # img  = blend_background(img)
                img  = finalize(img)
                arr  = np.array(img, dtype=np.uint8)
                writer.write(serialize_example(arr, word))
        print(f"✅ shard written: {path}")

# اجرا
# if __name__ == "__main__":
#     write_sharded_tfrecords()
