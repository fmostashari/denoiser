from moviepy.editor import VideoFileClip
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

#((hop_length/n-fft)-1)*100=87.5
#binary matrix/array
#hard binary mask
# تغییر ناگهانی در دامنه باعث ایجاد صدای تیک در باز سازی میشود
# برای رفع این مشکل میتوان هم پوشانی پنجره ها را بیشتر کرد که باعث افزاییش حجم محاسبات میشود
#sr 22050

n_fft = 2048
hop_length = 256
noise_duration = 1.5
n_std_thresh = 1.2

input_path = "C:/Users/TAMIRLAND/Downloads/test2.wav"
output_path = "C:/Users/TAMIRLAND/Downloads/spectral_gating_test2_clean.wav"

def reduce_noise(y, sr, n_fft=2048, hop_length=256, n_std_thresh=1.2, noise_duration=0.5):
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)

    noise_frames = magnitude[:, :int(noise_duration * sr / hop_length)] # 1.5 * 22050 = 33075/256=129
    noise_mean = np.mean(noise_frames, axis=1, keepdims=True)#time column
    noise_std = np.std(noise_frames, axis=1, keepdims=True)#standard deviation
    noise_thresh = noise_mean + n_std_thresh * noise_std #less x

    mask = magnitude >= noise_thresh
    filtered_magnitude = magnitude * mask
    filtered_stft = filtered_magnitude * np.exp(1j * phase)
    filtered_audio = librosa.istft(filtered_stft, hop_length=hop_length)

    return filtered_audio, stft, filtered_stft, mask

y, sr = librosa.load(input_path, sr=None)
filtered_y, stft_before, stft_after, mask = reduce_noise(y, sr,
                                                        n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        noise_duration=noise_duration,
                                                        n_std_thresh=n_std_thresh)

sf.write(output_path, filtered_y, sr)
print("File saved:", output_path)


plt.figure(figsize=(18, 10))
plt.subplot(2, 2, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_before), ref=np.max),
                         sr=sr, hop_length=hop_length, y_axis='log', x_axis='time', fmin=20, fmax=4000)
plt.title('Before Noise Reduction')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 2, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_after), ref=np.max),
                         sr=sr, hop_length=hop_length, y_axis='log', x_axis='time', fmin=20, fmax=4000)
plt.title('After Noise Reduction')
plt.colorbar(format='%+2.0f dB')

plt.subplot(2, 2, 3)
librosa.display.specshow(mask.astype(float), sr=sr, hop_length=hop_length, y_axis='log', x_axis='time', fmin=20, fmax=4000)
plt.title('Binary Mask')
plt.colorbar(label='0 (Noise) or 1 (Signal)')


frame_idx = 100  
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
magnitude_frame = np.abs(stft_before[:, frame_idx])
plt.subplot(2, 2, 4)
plt.plot(freqs, magnitude_frame)
plt.title(f'Frequency Spectrum (Frame {frame_idx})')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(20, 4000)  
plt.grid(True)

plt.tight_layout()
plt.show()