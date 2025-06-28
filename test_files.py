import tensorflow as tf

def parse_function(filepath):
    try:
        image = tf.io.read_file(filepath)
        image_decoded = tf.image.decode_png(image, channels=3)
        return image_decoded
    except:
        # برای تشخیص داده‌های مشکل‌دار
        print(f"Invalid image file: {filepath}")
        return None

filepaths = [...]  # لیست مسیرهای تصاویر
dataset = tf.data.Dataset.from_tensor_slices(filepaths)
dataset = dataset.map(parse_function).filter(lambda x: x is not None)