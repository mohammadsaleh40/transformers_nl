import os
import random
from PIL import Image, ImageDraw, ImageFont

def generate_digit_dataset(
    digits,
    font_paths,
    output_dir="digit_dataset",
    image_size=(256, 256),
    font_size_range=(3, 15),
    margin=3,
    images_per_digit=100,
    max_retries=100
):
    """
    تولید دیتاست تصویری اعداد با تنوع زیاد و نام‌گذاری استاندارد
    """
    os.makedirs(output_dir, exist_ok=True)

    # ایجاد پوشه‌های جداگانه برای هر عدد
    digit_dirs = {}
    for digit in digits:
        digit_dir = os.path.join(output_dir, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
        digit_dirs[digit] = digit_dir

    # بارگذاری فونت‌ها
    fonts = []
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, size=font_size_range[0])
            fonts.append(font)
        except Exception as e:
            print(f"فونت '{path}' نامعتبر است: {e}")
    if not fonts:
        raise FileNotFoundError("هیچ فونت معتبری یافت نشد!")

    # تولید تصاویر برای هر عدد
    for digit in digits:
        for idx in range(images_per_digit):
            attempt = 0
            success = False

            while not success and attempt < max_retries:
                try:
                    # انتخاب تصادفی فونت و اندازه
                    font = random.choice(fonts)
                    font_size = random.randint(*font_size_range)
                    current_font = font.font_variant(size=font_size)

                    # محاسبه اندازه متن
                    dummy_img = Image.new('RGB', (1, 1))
                    draw_dummy = ImageDraw.Draw(dummy_img)
                    text_bbox = draw_dummy.textbbox((0, 0), digit, font=current_font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    # ایجاد تصویر موقت با متن
                    temp_size = (text_width + 50, text_height + 50)
                    temp_img = Image.new('RGBA', temp_size, (0, 0, 0, 0))
                    draw_temp = ImageDraw.Draw(temp_img)
                    draw_temp.text((temp_size[0] // 2, temp_size[1] // 2), digit,
                                   font=current_font, fill=(0, 0, 0, 255))

                    # چرخش تصویر
                    angle = random.randint(-45, 45)
                    rotated_img = temp_img.rotate(angle, expand=True)

                    # ایجاد تصویر اصلی
                    main_img = Image.new('RGB', image_size, color='white')

                    # محاسبه موقعیت قرارگیری
                    max_x = image_size[0] - rotated_img.width - margin
                    max_y = image_size[1] - rotated_img.height - margin

                    if max_x <= margin or max_y <= margin:
                        raise ValueError("تصویر چرخانده شده بزرگتر از حاشیه است")

                    x = random.randint(margin, max_x)
                    y = random.randint(margin, max_y)

                    # چسباندن تصویر چرخانده شده
                    main_img.paste(rotated_img, (x, y), rotated_img)

                    # ذخیره تصویر
                    filename = f"{digit}_{idx}.png"
                    output_path = os.path.join(output_dir, filename)
                    main_img.save(output_path)
                    print(f"ذخیره شد: {output_path}")
                    success = True

                except Exception as e:
                    # نوشتن دلیل خطا
                    print(f"خطا در نوشتن دلیل خطا: {e}")
                    
                    attempt += 1
                    print(f"تلاش مجدد برای {digit}_{idx} (خطا: {e})")

            if not success:
                print(f"عدم موفقیت در تولید {digit}_{idx} پس از {max_retries} تلاش")


# 🔧 تنظیمات اولیه
if __name__ == "__main__":
    digits = [str(i) for i in range(10)]  # ['0', '1', ..., '9']
    font_paths = [
        "danstevis.otf",
        "Amiri Quran Colored.ttf",
        "Far.Yas.ttf",
        "B Moj.ttf",
        "A Iranian Sans.ttf"
        
    ]

    generate_digit_dataset(
        digits=digits,
        font_paths=font_paths,
        output_dir="digit_dataset",
        image_size=(128, 128),
        font_size_range=(40, 60),
        margin=15,
        images_per_digit=1000,
        max_retries=50
    )

# %%

# %%
