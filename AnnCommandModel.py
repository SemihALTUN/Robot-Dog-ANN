import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from PIL import Image

# Veri Seti Hazırlığı
data_dir = r"datasetcommand"  # Veri seti klasörü
spectrogram_dir = "spectrograms"  # Kaydedilecek mel-spektrogramlar

# Ses verisine gürültü ekleme
def add_noise(data, noise_factor=0.03):
    noise = noise_factor * np.random.randn(len(data))
    augmented_data = data + noise
    return np.clip(augmented_data, -1.0, 1.0)

# Ses verisinin pitch'ini değiştirme
def adjust_pitch(data, sr, n_steps=2):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

# Ses hızını değiştirme
def adjust_speed(data, speed_factor=1.2):
    return librosa.effects.time_stretch(data, rate=speed_factor)

# Komut etiketlerini tanımla (şarkıcı yerine komutlar)
commands = ["dur", "geri_git", "ileri_git", "saga_dogru_git", "sola_dogru_git"]
command_labels = {command: idx for idx, command in enumerate(commands)}  # Komut etiketlerini sözlük şeklinde tanımlıyoruz

# Veri arttırma teknikleri
augmentations = {
    "normal": [],  # Orijinal veri
    "noise": [add_noise],  # arka plan gürültü ekleme
    "pitch_up": [lambda y, sr: adjust_pitch(y, sr, n_steps=2)],  # pitch arttırma
    "pitch_down": [lambda y, sr: adjust_pitch(y, sr, n_steps=-2)],  # pitch azaltma
    "speed": [lambda y: adjust_speed(y, speed_factor=1.2)]  # hız arttırma
}

# Mel-Spektrogram çıkarma ve kaydetme
def create_spectrogram(file_path, save_path, augmentations=None):
    """Ses dosyasından mel-spektrogram çıkar ve kaydet."""
    # Komut adını spectrogram dosya yolundan çıkart
    command_folder = os.path.basename(os.path.dirname(save_path))  # Komut adını çıkart
    spectrogram_folder = os.path.join(spectrogram_dir, command_folder)  # Komuta ait klasör
    os.makedirs(spectrogram_folder, exist_ok=True)  # Komut klasörü yoksa oluştur

    spectrogram_path = os.path.join(spectrogram_folder, os.path.basename(save_path))  # Dosya yolu oluştur
    if not os.path.exists(spectrogram_path):  # Eğer spektrogram daha önce oluşturulmamışsa
        y, sr = librosa.load(file_path, duration=5)

        # Uygulanacak augmentations listesi
        if augmentations:
            for augment in augmentations:
                if callable(augment):
                    if 'sr' in augment.__code__.co_varnames:
                        y = augment(y, sr)
                    else:
                        y = augment(y)

        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        plt.figure(figsize=(2.56, 2.56))  # 128x128 pixel görüntü
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max),
                                 y_axis='mel', fmax=8000, x_axis='time')
        plt.axis('off')
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close()

# Veri seti hazırlığında, spectrogramları doğru klasöre kaydedecek şekilde düzenleme
data = []  # Mel spektrogram dosya yolu
command_labels_list = []  # Mel spektrogram hangi komuta ait

for command in tqdm(commands, desc="Processing Command"):  # her bir komut dosyasını gezme
    command_label = command_labels[command]  # oluşturulan etiketler ile komut isimlerini eşleme
    command_dir = os.path.join(data_dir, command)  # Komut klasöründeki dosyalar
    for file in tqdm(os.listdir(command_dir), desc=f"Processing {command}", leave=False):  # Komut klasöründeki şarkıları listeler
        file_path = os.path.join(command_dir, file)  # Komuta ait dosya yolu oluşturur
        for aug_type, aug_list in augmentations.items():
            # Dosya adı komut adıyla birlikte, augmentasyon adıyla isimlendir
            spectrogram_path = os.path.join(spectrogram_dir, command, f"{command}_{file}_{aug_type}.png")
            create_spectrogram(file_path, spectrogram_path, augmentations=aug_list)  # Spektrogramı oluştur
            data.append(spectrogram_path)  # Dosya yolunu kaydet
            command_labels_list.append(command_label)  # Etiketleri kaydet

# 2. Veriyi Bölme
X_train, X_test, y_train, y_test = train_test_split(
    data, command_labels_list, test_size=0.2, random_state=42
)

# Görüntüleri yükleme ve normalizasyon
def load_image(image_path):
    """Mel-spektrogram görüntüsünü yükler ve normalize eder."""
    img = Image.open(image_path).convert('L')  # Gri tonlamaya çevir
    img = img.resize((128, 128))  # 128x128 boyutuna çevir
    return np.array(img) / 255.0

X_train = np.array([load_image(path) for path in X_train]).reshape(-1, 128, 128, 1)
X_test = np.array([load_image(path) for path in X_test]).reshape(-1, 128, 128, 1)
y_train = to_categorical(y_train, num_classes=len(command_labels))
y_test = to_categorical(y_test, num_classes=len(command_labels))

# 3. İleri Yönlü Yapay Sinir Ağı Modeli
model = Sequential([
    Flatten(input_shape=(128, 128, 1)),
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),  # Dropout ekledik
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),  # Dropout ekledik
    Dense(len(command_labels), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 4. Modeli Eğitme
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50, batch_size=32
)

# 5. Modeli Değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Modeli Kaydet
model.save(f"simple_ann_model_with_augmentation_semih_Deneme{test_accuracy:.2f}.h5")

# 6. Eğitim Grafikleri Çizme
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Eğitim doğruluğu ve kaybı grafikleri
plt.figure(figsize=(12, 6))

# Doğruluk Grafiği
plt.subplot(1, 2, 1)
plt.plot(accuracy, label='Eğitim Doğruluğu')
plt.plot(val_accuracy, label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

# Kayıp Grafiği
plt.subplot(1, 2, 2)
plt.plot(loss, label='Eğitim Kayıp')
plt.plot(val_loss, label='Doğrulama Kayıp')
plt.title('Eğitim ve Doğrulama Kayıp')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.tight_layout()
plt.show()
