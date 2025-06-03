# %%
import tensorflow as tf
import os
# %%
# 1. پیش‌پردازش داده‌ها
def preprocess_image(path):
    # استخراج لیبل
    filename = tf.strings.split(path, os.path.sep)[-1]
    label = tf.strings.split(filename, '_')[0]
    label = tf.strings.to_number(label, out_type=tf.int32)

    # خواندن و پردازش تصویر
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = tf.image.rgb_to_grayscale(img)
    img = img / 255.0  # نرمال‌سازی

    return img, label
# --- Hyperparameters ---
BATCH_SIZE = 256                # اندازه بچ (قابل تغییر)
LEARNING_RATE = 3e-4           # نرخ یادگیری
EPOCHS = 50                    # تعداد ایپاک‌ها
EARLY_STOPPING_PATIENCE = 10   # تعداد دوره‌های صبر برای Early Stopping
REDUCE_LR_PATIENCE = 5         # تعداد دوره‌های صبر برای ReduceLROnPlateau
# -----------------------
# 2. بارگذاری داده‌ها
image_paths = tf.data.Dataset.list_files('digit_dataset/*.png', shuffle=True)
dataset = image_paths.map(preprocess_image)

# تقسیم داده‌ها به train/test
train_size = int(0.8 * 10000)  # 8000 تصویر برای آموزش
val_size = 10000 - train_size  # 2000 تصویر برای اعتبارسنجی
train_dataset = dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 3. پیاده‌سازی Vision Transformer
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
# این مدل به دقت ۱۰۰ درصدی رسید.
def create_cnn_model(input_shape=(128, 128, 1), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_hybrid_model(input_shape=(128, 128, 1), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)

    # بخش CNN: استخراج ویژگی
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    # تبدیل خروجی CNN به پچ‌های قابل فهم برای Transformer
    _, h, w, c = x.shape
    num_patches = h * w
    patch_dim = c
    x = tf.keras.layers.Reshape((num_patches, patch_dim))(x)  # شکل: (batch, 256, 128)

    # کدگذاری پچ‌ها (همانند ViT)
    x = PatchEncoder(num_patches=num_patches, projection_dim=patch_dim)(x)

    # لایه‌های Transformer
    for _ in range(2):   # فقط 4 لایه Transformer (به جای 12)
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=patch_dim // 4, dropout=0.1
        )(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, x])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = tf.keras.layers.Dense(256, activation='relu')(x3)
        x3 = tf.keras.layers.Dense(patch_dim)(x3)
        x = tf.keras.layers.Add()([x3, x2])

    # خروجی نهایی
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
# %%
# 4. ساخت و کامپایل مدل
vit_model = create_cnn_model()
vit_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# کال‌بک‌های بهتر
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]
# %%
import matplotlib.pyplot as plt

# نمایش یک تصویر تصادفی
sample_image = tf.io.read_file('digit_dataset/0_0.png')
sample_image = tf.io.decode_png(sample_image, channels=3)
plt.imshow(sample_image.numpy())
plt.title("Sample Image")
plt.show()
# %%

# %%
# 5. آموزش مدل
vit_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ModelCheckpoint('best_model.keras')
    ]
)

# %%
loss, accuracy = vit_model.evaluate(val_dataset)
print(f"Validation Accuracy: {accuracy:.2f}")
# %%
# 6. مثال استفاده از مدل
def predict_digit(image_path):
    img = preprocess_image(image_path)[0]  # فقط تصویر
    prediction = vit_model.predict(tf.expand_dims(img, axis=0))
    return tf.argmax(prediction[0]).numpy()

# تست یک تصویر
test_image = "digit_dataset/7_22.png"
predicted_digit = predict_digit(test_image)
print(f"عدد تشخیص داده شده: {predicted_digit}")
# %%

# %%
# %%
