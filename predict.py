
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import youtube_dl
from os import path, listdir
from dataset import get_track_names
from convtasnetparam import get_param
from convtasnet import ConvTasNet
from loss import SISNR, SDR

MAX_EPOCH = 100
LEARNING_RATE = 1e-3
EPSILION = 1e-8

param = get_param(eps=EPSILION)

adam = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# model = ConvTasNet.make(param, adam, SISNR(eps=EPSILION))
# directory_name = f"/home/kaparoo/Conv-TasNet/convtasnet_train/training_sisnr_overlap_{param.causality}_{param.gating}_{param.T_hat}_{param.C}_{param.N}_{param.L}_{param.B}_{param.Sc}_{param.H}_{param.P}_{param.X}_{param.R}"
model = ConvTasNet.make(param, adam, SDR(eps=EPSILION))
directory_name = f"/home/kaparoo/Conv-TasNet/convtasnet_train/training_sdr_overlap_{param.causality}_{param.gating}_{param.T_hat}_{param.C}_{param.N}_{param.L}_{param.B}_{param.Sc}_{param.H}_{param.P}_{param.X}_{param.R}"

if path.exists(directory_name):
    checkpoints = [name for name in listdir(
        directory_name) if "ckpt" in name]
    checkpoints.sort()
    checkpoint_name = checkpoints[-1].split(".")[0]
    model.load_weights(f"{directory_name}/{checkpoint_name}.ckpt")


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
# TODO | must fix num_samples considering overlapping
num_samples = audio.shape[0]
num_portions = num_samples // (param.T_hat * param.L)
num_samples = num_portions * (param.T_hat * param.L)

audio = audio[:num_samples]
audio = np.reshape(audio, (num_portions, param.T_hat, param.L))

print("predicting...")

separated = model.predict(audio)
separated = np.transpose(separated, (1, 0, 2, 3))
separated = np.reshape(separated, (param.C, num_samples))

print("saving...")

for idx, track in enumerate(get_track_names()):
    sf.write(
        f"/home/kaparoo/Conv-TasNet/convtasnet_predict/{title}_{track}.wav", separated[idx], sr)
