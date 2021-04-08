import tensorflow as tf
from os import path, listdir
from conv_tasnet import SDR
from conv_tasnet import SISNR
from conv_tasnet import ConvTasNet
from conv_tasnet import ConvTasNetParam
from musdb_dataset import make_dataset, get_track_names
import matplotlib.pyplot as plt


def get_directory_name(use_sdr: bool, param: ConvTasNetParam,
                       prefix: str = "/home/kaparoo/conv-tasnet/") -> str:
    """Make directory name from hyperparameters.

    Args:
        param (ConvTasNetParam): Hyperparameters container
        use_sdr (bool): Flag corresponding to the loss function of the model
        prefix (str): Directory prefix

    Returns:
        directory_name (str): Directory name
    """
    p = param  # alias
    loss_str = "use_sdr" if use_sdr else "sisnr"
    caual_str = "causal" if p.causal else "noncausal"
    directory_name = f"{prefix}train_results/{loss_str}_{caual_str}_K{p.K}_C{p.C}_L{p.L}_N{p.N}_B{p.B}_S{p.S}_H{p.H}_P{p.P}_X{p.X}_R{p.R}"
    return directory_name
# get_directory_name(*) end


def make_model(learning_rate: float = 0.001, l2_norm_clip: float = 5,
               eps: float = 1e-8, use_sdr: bool = False, causal: bool = True,
               num_class: int = 3, dir_prefix: str = "/home/kaparoo/conv-tasnet/"):
    """Make Conv-TasNet Instance.

    Args:
        learning_rate (int): Learning rate
        l2_norm_clip (float): Gradient clipping factor
        eps (float): Small constant for numerical stability
        use_sdr (bool): Metric flag
        causal (bool): Caulity flag
        num_class (int): Numbers of class
        dir_prefix (str): Directory prefix

    Returns:
        model (ConTasNet): Model instance
        param (ConvTasNetParam): Hyperparameters' container
        dir_name (str): directory name
    """
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                    clipnorm=l2_norm_clip)
    loss = SDR(eps=eps) if use_sdr else SISNR(eps=eps)
    param = ConvTasNetParam(C=num_class, causal=causal, eps=eps)
    dir_name = get_directory_name(use_sdr, param, dir_prefix)
    model = ConvTasNet.make(param, adam, loss)
    return model, param, dir_name
# make_model(*) end


def train_model(max_epoch: int = 100, causal: bool = True, use_sdr: bool = False, eps: float = 1e-8,
                num_class: int = 3, dir_prefix: str = "/home/kaparoo/conv-tasnet/"):
    """Model trainging

    Args:
        max_epoch (int): Maximum epoch for model training
        causal (bool): Causality flag
        use_sdr (bool): Loss floag
            - True: SDR(eps)
            - False: SISNR(eps)
        eps (float): Small constant for numerical stability
        prefix (str): Directory prefix

    Returns:
        history (tf.keras.callbacks.History): History of model loss and accuracy
    """
    model, param, dir_name = make_model(num_class=num_class,
                                        causal=causal,
                                        use_sdr=use_sdr,
                                        dir_prefix=dir_prefix)
    model.summary()
    history = None
    epoch = 0
    if path.exists(dir_name):
        checkpoints = [name for name in listdir(dir_name) if "ckpt" in name]
        checkpoints.sort()
        checkpoint_name = checkpoints[-1].split(".")[0]
        epoch = int(checkpoint_name) + 1
        model.load_weights(f"{dir_name}/{checkpoint_name}.ckpt")

    while epoch < max_epoch:
        checkpoint_path = f"{dir_name}/{epoch:05d}.ckpt"
        print("Checkpoint path: ", checkpoint_path)
        print(f"Epoch: {epoch}")
        dataset = make_dataset(param, 5, 100, 1000)
        history = model.fit(dataset)
        model.save_weights(checkpoint_path)
        epoch += 1

    return history
# train_modle(*) end


if __name__ == "__main__":
    max_epoch = 50
    causal = False
    use_sdr = True
    num_track = len(get_track_names())
    history = train_model(max_epoch=max_epoch,
                          causal=causal,
                          use_sdr=use_sdr,
                          num_class=num_track)
