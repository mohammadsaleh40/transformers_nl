# %%
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers , Model
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
class LayerScale(tf.keras.layers.Layer):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = tf.keras.initializers.Constant(init_value)(shape=[dim])

    def call(self, x):
        return x * self.gamma
# --- Hyperparameters ---
BATCH_SIZE = 128                # اندازه بچ (قابل تغییر)
LEARNING_RATE = 1e-4           # نرخ یادگیری
EPOCHS = 40                    # تعداد ایپاک‌ها
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
# %%
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

    # CNN قوی‌تر
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # خروجی: 16×16×64

    # تبدیل به پچ‌ها
    _, h, w, c = x.shape
    num_patches = h * w
    x = tf.keras.layers.Reshape((num_patches, c))(x)  # (batch, 256, 64)

    # Patch Encoder با ابعاد بالاتر
    x = PatchEncoder(num_patches=num_patches, projection_dim=64)(x)
    # کاهش لایه‌های Transformer
    # درون مدل هیبریدی:
    for _ in range(4):  # افزایش تعداد لایه‌های Transformer
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=8, dropout=0.1
        )(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, x])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = tf.keras.layers.Dense(256, activation='relu')(x3)
        x3 = tf.keras.layers.Dropout(0.3)(x3)
        x3 = tf.keras.layers.Dense(64)(x3)
        x3 = LayerScale(64)(x3) 
        x = tf.keras.layers.Add()([x3, x2])

    # خروجی نهایی
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# %%

def transformer_block(x, num_heads=8, key_dim=32):
    """بلوک ترنسفورمر با لایه‌های خودتوجهی"""
    # Layer normalization اول
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # توجه چندهدفه
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
    )(x1, x1)
    
    # اتصال باقیمانده
    x2 = layers.Add()([attention_output, x])
    
    # Layer normalization دوم
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    
    # MLP
    x3 = layers.Dense(256, activation='gelu')(x3)
    x3 = layers.Dropout(0.2)(x3)
    x3 = layers.Dense(x.shape[-1])(x3)
    
    # اتصال باقیمانده نهایی
    return layers.Add()([x3, x2])

def create_model(input_shape=(128, 128, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    
    # لایه‌های اولیه CNN
    x = inputs
    x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((4, 4))(x)  # (32, 32, 8)
    x = layers.Conv2D(4, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)  # (32, 32, 4)
    x = layers.MaxPooling2D((4, 4))(x)  # (8, 8, 4)
    
    # توکنایز 4x4
    patches = layers.Conv2D(
        filters=64,
        kernel_size=4,
        strides=4,
        padding='valid'
    )(x)  # (2, 2, 64) - چون (8/4=2)
    
    patches = layers.Reshape((4, 64))(patches)  # 4 پچ 64-بعدی
    
    # افزودن موقعیت
    positions = tf.range(start=0, limit=4, delta=1)
    position_embedding = layers.Embedding(
        input_dim=4,
        output_dim=64
    )(positions)
    x_att = patches + position_embedding
    
    # دو لایه ترنسفورمر
    for _ in range(4):
        x_att = transformer_block(x_att)  # شکل خروجی: (4, 64)
    
    # تبدیل به فرمت تصویری
    x_att = layers.Reshape((2, 2, 64))(x_att)  # (2, 2, 64)
    
    # کاهش ابعاد داده اولیه
    x_orig = layers.Conv2D(64, (3, 3), strides=16, padding='same')(inputs)  # (8, 8, 64)
    x_orig = layers.BatchNormalization()(x_orig)
    
    # تطابق ابعاد با upsampling
    x_att = layers.UpSampling2D(size=(4, 4))(x_att)  # (8, 8, 64)
    
    # ترکیب ویژگی‌ها
    combined = layers.Concatenate(axis=-1)([x_att, x_orig])  # (8, 8, 128)
    
    # سه لایه CNN
    x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)  # (8, 8, 32)
    
    # طبقه‌بندی
    x = layers.GlobalAveragePooling2D()(x)  # (32,)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)


# %%
# 4. ساخت و کامپایل مدل
vit_model = create_model()
vit_model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),  # AdamW به جای Adam
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Data Augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.1)
])

train_dataset = dataset.take(train_size).shuffle(1000).batch(BATCH_SIZE).map(
    lambda x, y: (augmentation(x, training=True), y)
).prefetch(tf.data.AUTOTUNE)
# کال‌بک‌های بهتر
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]
# %%
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
    callbacks=callbacks,
)

# %%
loss, accuracy = vit_model.evaluate(val_dataset)
print(f"Validation Accuracy: {accuracy:.2f}")
# %%
# 6. مثال استفاده از مدل

# %%
def predict_digit(image_path):
    """Predict digit in image and return both prediction and original image"""
    try:
        img = preprocess_image(image_path)[0]  # فقط تصویر
        prediction = vit_model.predict(tf.expand_dims(img, axis=0), verbose=0)
        predicted_digit = tf.argmax(prediction[0]).numpy()
        return predicted_digit, img.numpy().squeeze()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def visualize_predictions():
    """Visualize model predictions in a 10x10 grid"""
    plt.figure(figsize=(15, 15))
    plt.suptitle("Model Predictions Visualization\n(Green: Correct, Red: Incorrect)", fontsize=16)
    
    for i in range(10):  # 10 rows for digits 0-9
        for j in range(10):  # 10 columns for samples
            # Create subplot
            plt.subplot(10, 10, i*10 + j + 1)
            
            # Generate random image path
            rand_idx = np.random.randint(0, 1000)
            test_image = f"digit_dataset/{i}_{rand_idx}.png"
            
            # Get prediction
            prediction, img = predict_digit(test_image)
            
            if prediction is not None and img is not None:
                # Display image
                plt.imshow(img, cmap='gray')
                
                # Color coding for correct/incorrect predictions
                true_label = i
                color = 'green' if prediction == true_label else 'red'
                
                # Show prediction with confidence
                plt.title(f"Pred: {prediction}\nTrue: {true_label}", 
                         color=color, fontsize=10)
            else:
                # Show error message if prediction failed
                plt.text(0.5, 0.5, "Error", ha='center', va='center')
            
            plt.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("digit_predictions_grid.png", dpi=300, bbox_inches='tight')
    plt.show()

# Run visualization
visualize_predictions()
# %%
for name, var in vit_model.named_variables():
    print(f"{name}: {var.numpy().mean():.4f}")
# %%
