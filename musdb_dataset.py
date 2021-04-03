import tensorflow as tf
import numpy as np
import musdb
import random
from tqdm import tqdm
from conv_tasnet import ConvTasNet, ConvTasNetParam

TRACK_NAMES = ("vocals", "drums", "bass")


def get_track_names():
    return TRACK_NAMES


def decode_source(track):
    rtn = {"audio": (track.audio[:, 0], track.audio[:, 1])}
    rtn["length"] = rtn["audio"][0].shape[-1]
    for target in get_track_names():
        audio = track.targets[target].audio
        rtn[target] = (audio[:, 0], audio[:, 1])
    return rtn


def musdb_generator(param: ConvTasNetParam, num_songs: int, batch_size: int,
                    n: int, musdb_dir: str = "/home/kaparoo/musdb18"):
    db = list(musdb.DB(root=musdb_dir, subsets="train").tracks)
    random.shuffle(db)
    db = db[:num_songs]

    print("Decoding dataset...")
    mus = []
    for track in tqdm(db):
        mus.append(decode_source(track))
    print("Decoding dataset...done")

    duration = (param.T_hat + 1) * param.L // 2  # 50% overlap
    for _ in range(n):
        X = []
        Y = []
        prev_batch_start = []
        for _ in range(batch_size):
            track = random.choice(mus)

            batch_start = random.randint(0, track["length"] - duration)
            while batch_start in prev_batch_start:
                batch_start = random.randint(0, track["length"] - duration)
            prev_batch_start.append(batch_start)

            x_0, x_1, y_0, y_1 = [], [], [], []

            _start = batch_start
            for _ in range(param.T_hat):
                # [0], [1] for stereo
                _end = _start + param.L
                segment_0 = np.array(track["audio"][0][_start:_end])  # 1 x L
                segment_1 = np.array(track["audio"][1][_start:_end])  # 1 x L
                sources_0 = [track[target][0][_start:_end]
                             for target in get_track_names()]  # C x 1 x L
                sources_1 = [track[target][1][_start:_end]
                             for target in get_track_names()]  # C x 1 x L
                _start = _start + (param.L // 2)  # 50% overlap
                x_0.append(segment_0)
                x_1.append(segment_1)
                y_0.append(sources_0)
                y_1.append(sources_1)

            x_0 = np.array(x_0)
            x_1 = np.array(x_1)
            y_0 = np.array(y_0)
            y_1 = np.array(y_1)

            y_0 = y_0.transpose((1, 0, 2))
            y_1 = y_1.transpose((1, 0, 2))

            X.extend([x_0, x_1])
            Y.extend([y_0, y_1])

        yield np.array(X), np.array(Y)


def make_dataset(param: ConvTasNetParam, num_songs: int, batch_size: int, n: int):
    return tf.data.Dataset.from_generator(lambda: musdb_generator(param, num_songs, batch_size, n),
                                          output_types=(
                                              tf.float32, tf.float32),
                                          output_shapes=(tf.TensorShape((batch_size * 2, param.T_hat, param.L)),
                                                         tf.TensorShape((batch_size * 2, param.C, param.T_hat, param.L))))
