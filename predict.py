import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import youtube_dl
from os import path, listdir
from train import make_model
from musdb_dataset import get_track_names

# make model with hyperparameters that specified in train.py
model, param, dir_name = make_model()

if path.exists(dir_name):
    checkpoints = [name for name in listdir(dir_name) if "ckpt" in name]
    checkpoints.sort()
    checkpoint_name = checkpoints[-1].split(".")[0]
    model.load_weights(f"{dir_name}/{checkpoint_name}.ckpt")


def youtube_dl_hook(d):
    if d["status"] == "finished":
        print("Done downloading...")


url = "gdZLi9oWNZg"
ydl_opts = {
    "format": "bestaudio/best",
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
        "preferredquality": "44100",
    }],
    "outtmpl": "%(title)s.wav",
    "progress_hooks": [youtube_dl_hook],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    status = ydl.download([url])

title = info.get("title", None)
filename = title + ".wav"
audio, sr = librosa.load(filename, sr=44100, mono=True)
num_samples = audio.shape[0]
num_portions = num_samples // (param.K * param.L)
num_samples = num_portions * (param.K * param.L)

audio = audio[:num_samples]
audio = np.reshape(audio, (num_portions, param.K, param.L))

print("predicting...")

separated = model.predict(audio)
separated = np.transpose(separated, (1, 0, 2, 3))
separated = np.reshape(separated, (param.C, num_samples))

print("saving...")

for idx, track in enumerate(get_track_names()):
    sf.write(
        f"/home/kaparoo/conv-tasnet/predict_results/{title}_{track}.wav", separated[idx], sr)
