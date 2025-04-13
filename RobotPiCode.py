from tensorflow.keras.models import load_model
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time
from adafruit_servokit import ServoKit
import os
from PIL import Image

# Kişi doğrulama modeli
verification_model = load_model(r"/home/rasperrypi/Desktop/insan_tanima_modeli.h5")

# Komut modelini yükle
model = load_model(r"/home/rasperrypi/Desktop/simple_ann_model_with_augmentation_semih_Deneme0.70.h5")

# ServoKit
kit = ServoKit(channels=16)

# Komut etiketleri
command_labels = {
    'dur': 0,
    'geri_git': 1,
    'ileri_git': 2,
    'saga_dogru_git': 3,
    'sola_dogru_git': 4
}

# Komut çıktıları
command_outputs = {
    'dur': [0, 0, 0, 0],
    'geri_git': [0, 1, 1, 0],
    'ileri_git': [1, 0, 0, 1],
    'saga_dogru_git': [0, 0, 1, 1],
    'sola_dogru_git': [1, 1, 0, 0]
}


def record_audio(duration=5, fs=22050, filename="mic_input.wav"):
    print(f"🎤 {duration} saniye boyunca konuşun...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = np.squeeze(audio)
    write(filename, fs, (audio * 32767).astype(np.int16))
    print("✅ Ses kaydedildi.")
    return filename


def extract_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, duration=5)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.figure(figsize=(2.56, 2.56))
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max),
                             y_axis='mel', fmax=8000, x_axis='time')
    plt.axis('off')
    temp_path = "temp_spectrogram.png"
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    img = Image.open(temp_path).convert('L')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 128, 128, 1)
    os.remove(temp_path)
    return img_array


def predict_person(audio_path):
    spectrogram = extract_spectrogram(audio_path)
    prediction = verification_model.predict(spectrogram)
    predicted_index = np.argmax(prediction)
    if predicted_index == 0:  # 0: 'Hüseyin'
        print("🧑‍🦱 Kişi tanındı: Hüseyin")
        return 1  # Hüseyin tanındı
    else:
        print("❌ Kişi tanınmadı.")
        return 2  # Kişi tanınmadı


def predict_command(audio_path):
    spectrogram = extract_spectrogram(audio_path)
    prediction = model.predict(spectrogram)
    predicted_index = np.argmax(prediction)
    predicted_label = list(command_labels.keys())[list(command_labels.values()).index(predicted_index)]
    output_bits = command_outputs[predicted_label]
    print(f"\n🎙️ Tahmin edilen komut: {predicted_label}")
    print(f"🔢 Komut çıktısı (4 bit): {output_bits}")
    return predicted_label, output_bits


# Komut fonksiyonları
def dur():
    print("🚦 Dur komutu uygulandı")
    kit.servo[11].angle = 100
    kit.servo[15].angle = 130
    kit.servo[7].angle = 70
    kit.servo[8].angle = 30
    kit.servo[12].angle = 120
    kit.servo[0].angle = 50
    kit.servo[3].angle = 60
    kit.servo[4].angle = 130
    time.sleep(3)


def ileri():
    print("🚶‍♂️ İleri komutu uygulandı")
    for _ in range(10):
        kit.servo[11].angle = 100
        kit.servo[7].angle = 40
        kit.servo[12].angle = 120
        kit.servo[3].angle = 110
        time.sleep(0.1)
        kit.servo[15].angle = 90
        kit.servo[8].angle = 120
        kit.servo[0].angle = 90
        kit.servo[4].angle = 40
        time.sleep(0.1)
        kit.servo[11].angle = 140
        kit.servo[7].angle = 70
        kit.servo[12].angle = 70
        kit.servo[3].angle = 60
        time.sleep(0.1)
        kit.servo[15].angle = 40
        kit.servo[8].angle = 70
        kit.servo[0].angle = 140
        kit.servo[4].angle = 90
        time.sleep(0.1)
    time.sleep(1)


def geri():
    print("⬅️ Geri git komutu uygulandı")
    for _ in range(10):
        kit.servo[11].angle = 130
        kit.servo[7].angle = 40
        kit.servo[12].angle = 90
        kit.servo[3].angle = 90
        time.sleep(0.1)
        kit.servo[15].angle = 100
        kit.servo[8].angle = 80
        kit.servo[0].angle = 110
        kit.servo[4].angle = 100
        time.sleep(0.1)
        kit.servo[11].angle = 100
        kit.servo[7].angle = 70
        kit.servo[12].angle = 120
        kit.servo[3].angle = 60
        time.sleep(0.1)
        kit.servo[15].angle = 80
        kit.servo[8].angle = 60
        kit.servo[0].angle = 70
        kit.servo[4].angle = 90
        time.sleep(0.1)


def saga_dogru_git():
    """Geri adımı YLine_Demo mantığıyla yapar."""
    print("⬅️ Sağa doru gitme komudu başlıyor")
    dur()
    for _ in range(10):
        kit.servo[3].angle = 20
        time.sleep(0.1)
        kit.servo[4].angle = 130
        time.sleep(0.1)
        kit.servo[3].angle = 60
        time.sleep(0.1)
        kit.servo[4].angle = 90
        time.sleep(0.1)

        kit.servo[11].angle = 60
        time.sleep(0.1)
        kit.servo[15].angle = 130
        time.sleep(0.1)
        kit.servo[11].angle = 100
        time.sleep(0.1)
        kit.servo[15].angle = 90
        time.sleep(0.1)

        kit.servo[3].angle = 20
        time.sleep(0.1)
        kit.servo[4].angle = 130
        time.sleep(0.1)
        kit.servo[3].angle = 60
        time.sleep(0.1)
        kit.servo[4].angle = 90
        time.sleep(0.1)

        kit.servo[11].angle = 60
        time.sleep(0.1)
        kit.servo[15].angle = 130
        time.sleep(0.1)
        kit.servo[11].angle = 100
        time.sleep(0.1)
        kit.servo[15].angle = 90
        time.sleep(0.1)


def sola_dogru_git():
    print("⬅️ Sola doğru git komutu uygulandı")
    for _ in range(10):
        kit.servo[11].angle = 130
        kit.servo[7].angle = 40
        kit.servo[12].angle = 90
        kit.servo[3].angle = 90
        time.sleep(0.1)
        kit.servo[15].angle = 100
        kit.servo[8].angle = 80
        kit.servo[0].angle = 110
        kit.servo[4].angle = 100
        time.sleep(0.1)


if __name__ == "__main__":
    while True:
        audio_file = record_audio()
        person_status = predict_person(audio_file)

        if person_status == 2:
            print("❌ Kişi tanınmadı. Programdan çıkılıyor.")
            break  # Kişi tanınmazsa programdan çıkılır

        # Eğer Hüseyin tanındıysa, komutları dinlemeye başla
        command, _ = predict_command(audio_file)

        if command == 'dur':
            dur()
        elif command == 'ileri_git':
            ileri()
        elif command == 'geri_git':
            geri()
        elif command == 'saga_dogru_git':
            saga_dogru_git()
        elif command == 'sola_dogru_git':
            sola_dogru_git()
        time.sleep(5)  # 1 saniye bekle