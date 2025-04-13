import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random

# Veri yollarını tanımlıyoruz
MY_VOICE_PATH = r"C:\Users\semii\OneDrive\Masaüstü\YSAegitim\Sesler\benim_sesim"  # Benim sesim klasörü
OTHER_VOICES_PATH = r"C:\Users\semii\OneDrive\Masaüstü\YSAegitim\Sesler\diger_sesler"  # Diğer sesler klasörü

CATEGORIES = ["benim_sesim", "diger_sesler"]  # Kategoriler


def extract_mfcc(file_path, max_pad_len=128):
    """Ses dosyasından MFCC özelliklerini çıkarır."""
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc  # 40 x 128 matris


# Verileri yükleme
X, y = [], []
for category in CATEGORIES:
    if category == "benim_sesim":
        folder_path = MY_VOICE_PATH  # Benim sesim klasörü
    else:
        folder_path = OTHER_VOICES_PATH  # Diğer sesler klasörü

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        mfcc = extract_mfcc(file_path)
        X.append(mfcc)
        y.append(category)

X = np.array(X)
y = np.array(y)

# Label encode
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize ve reshape (ANN için flatten yapılacak)
X = X / np.max(X)  # Normalize
X = X.reshape(X.shape[0], -1)  # (örnek_sayısı, 40*128)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Basit ANN modeli
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),  # Dropout oranını artırdık
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout oranını artırdık
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.5),  # Dropout oranını artırdık
    Dense(16, activation='relu'),
    Dense(len(CATEGORIES), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Erken durdurma callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# Modeli kaydet
model.save("insan_tanima_modeli.h5")
print("✅ Kişi tanıma modeli kaydedildi: person_tanima_modeli.h5")


# Eğitim sonuçlarını kaydet
def plot_and_save_history(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.title('Eğitim & Doğrulama Kaybı')
    plt.legend()
    plt.grid()
    plt.savefig("person_tanima_egitim_kaybi.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.title('Eğitim & Doğrulama Doğruluğu')
    plt.legend()
    plt.grid()
    plt.savefig("person_tanima_egitim_dogruluk.png")
    plt.close()


plot_and_save_history(history)
print("✅ Eğitim grafikleri kaydedildi: person_tanima_egitim_kaybi.png & person_tanima_egitim_dogruluk.png")

# Son doğruluk çıktısı
final_accuracy = history.history['val_accuracy'][-1] * 100
print(f"✅ En son doğrulama doğruluğu: %{final_accuracy:.2f}")

# Test verisinde modelin başarısı
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"🧪 Test Seti Doğruluğu: %{test_accuracy * 100:.2f}")


# Ses dosyasını yükleyip tahmin ettirme
def predict_sample(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = mfcc / np.max(mfcc)  # normalize
    mfcc = mfcc.reshape(1, -1)  # flatten + batch boyutu
    prediction = model.predict(mfcc)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    confidence = np.max(prediction) * 100
    return predicted_label[0], confidence


# Kullanıcıdan test sayısı alın
test_count = int(input("Kaç adet test yapmak istersiniz? "))

doğru_tahmin_sayisi = 0

# Test döngüsü
for _ in range(test_count):
    # Rastgele bir kategori ve dosya seçme
    random_category = random.choice(CATEGORIES)
    if random_category == "benim_sesim":
        test_folder = MY_VOICE_PATH  # Benim sesim klasörü
    else:
        test_folder = OTHER_VOICES_PATH  # Diğer sesler klasörü

    test_file = random.choice(os.listdir(test_folder))
    test_path = os.path.join(test_folder, test_file)

    # Model tahmini
    tahmin, guven = predict_sample(test_path)

    # Gerçek etiket ve tahmin karşılaştırması
    gerçek_etiket = random_category
    if tahmin == gerçek_etiket:
        doğru_tahmin_sayisi += 1

    # Sonuç yazdırma
    print(f"\n📁 Dosya: {test_file} ({gerçek_etiket})")
    print(f"🤖 Model Tahmini: {tahmin} (%{guven:.2f} güven)")

# Genel doğruluk oranını hesapla
doğruluk_oranı = (doğru_tahmin_sayisi / test_count) * 100
print(f"\n🧪 Genel Doğruluk Oranı: %{doğruluk_oranı:.2f}")
