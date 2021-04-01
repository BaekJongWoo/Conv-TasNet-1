import tensorflow as tf
import numpy as np
import musdb
import random
from tqdm import tqdm
from config import ConvTasNetParam
from convtasnet import ConvTasNet
from loss import SISNR, SDR


def get_track_names():
    return ("vocals", "drums", "bass")


def decode_source(track):
    rtn = {"audio": (track.audio[:, 0], track.audio[:, 1])}
    rtn["length"] = rtn["audio"][0].shape[-1]
    for target in get_track_names():
        audio = track.targets[target].audio
        rtn[target] = (audio[:, 0], audio[:, 1])
    return rtn


# TODO | must fix this function considering overlap
def musdb_generator(param: ConvTasNetParam, num_songs: int, batch_size: int, n: int,
                    musdb_dir: str = "/home/kaparoo/musdb18", overlap: int = 2):
    db = list(musdb.DB(root=musdb_dir, subsets="train").tracks)
    random.shuffle(db)
    db = db[:num_songs]

    print("Decoding dataset...")
    mus = []
    for track in tqdm(db):
        mus.append(decode_source(track))
    print("Decoding dataset...done")

    shape = (param.T_hat, param.L)
    duration = param.T_hat * param.L
    for _ in range(n):
        X = []
        Y = []
        for _ in range(batch_size):
            track = random.choice(mus)
            start = random.randint(0, track["length"] - duration)
            end = start + duration
            x_0 = np.reshape(track["audio"][0][start:end], shape)
            x_1 = np.reshape(track["audio"][1][start:end], shape)
            y_0 = [
                np.reshape(track[target][0][start:end], shape)
                for target in get_track_names()
            ]
            y_1 = [
                np.reshape(track[target][1][start:end], shape)
                for target in get_track_names()
            ]
            X.extend([x_0, x_1])
            Y.extend([y_0, y_1])
        yield np.array(X), np.array(Y)


def make_dataset(param: ConvTasNetParam, num_songs: int, batch_size: int, n: int):
    return tf.data.Dataset.from_generator(lambda: musdb_generator(param, num_songs, batch_size, n),
                                          output_types=(
                                              tf.float32, tf.float32),
                                          output_shapes=(tf.TensorShape((batch_size * 2, param.T_hat, param.L)),
                                                         tf.TensorShape((batch_size * 2, param.C, param.T_hat, param.L))))
