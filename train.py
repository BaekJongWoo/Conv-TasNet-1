import tensorflow as tf
from os import path, listdir
from conv_tasnet import SDR
from conv_tasnet import SISNR
from conv_tasnet import ConvTasNet
from conv_tasnet import ConvTasNetParam
from musdb_dataset import make_dataset


def get_directory_name(use_sdr: bool, param: ConvTasNetParam,
                       prefix: str = "/home/kaparoo/conv_tasnet") -> str:
    """Make directory name from hyperparameters

    Args:
        param (ConvTasNetParam): Hyperparameters container
        use_sdr (bool): Flag corresponding to the loss function of the model
        prefix (str): Directory prefix

    Returns:
        directory_name (str): Directory name
    """
    p = param  # alias
    loss_str = "sdr" if use_sdr else "sisnr"
    caual_str = "causal" if p.causal else "noncausal"
    directory_name = f"{prefix}/conv_tasnet_train/{loss_str}_{caual_str}_{p.K}K_{p.C}C_{p.L}L_{p.N}N_{p.B}B_{p.S}S_{p.H}H_{p.P}P_{p.X}X_{p.R}R"
    return directory_name
# get_directory_name(*) end


def make_model(learning_rate: float = 0.001, l2_norm_clip: float = 5,
               eps: float = 1e-8, use_sdr: bool = False, is_caual: bool = True,
               num_class: int = 4, dir_prefix: str = "/home/kaparoo/conv_tasnet"):
    """Make Conv-TasNet Instance

    Args:
        learning_rate (int): Learning rate
        l2_norm_clip (float): Gradient clipping factor
        eps (float): Small constant for numerical stability
        use_sdr (bool): Metric flag
        is_causal (bool): Caulity flag
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
    param = ConvTasNetParam(C=num_class, causal=is_caual, eps=eps)
    dir_name = get_directory_name(use_sdr, param, dir_prefix)
    model = ConvTasNet(param, adam, loss)
    return model, param, dir_name
# make_model(*) end


def train_model(max_epoch: int = 100, prefix: str = "/home/kaparoo/conv_tasnet"):
    """Model trainging

    Args:
        prefix (str): Directory prefix

    Returns:
        history
    """
    model, param, dir_name = make_model()
    history = None
    epoch = 0
    if path.exists(dir_name):
        checkpoints = [name for name in listdir(dir_name) if "ckpt" in name]
        checkpoints.sort()
        checkpoint_name = checkpoints[-1].split(".")[0]
        epoch = int(checkpoint_name) + 1
        model.load_weights(f"{dir_name}/{checkpoint_name}.ckpt")
    else:
        print(f"directory: `{dir_name}` is not exit!")
        return

    while epoch < max_epoch:
        checkpoint_path = f"{dir_name}/{epoch:05d}.ckpt"
        print(f"Epoch: {epoch}")
        dataset = make_dataset(param, 5, 100, 1000)
        history = model.fit(dataset)
        model.save_weights(checkpoint_path)
        epoch += 1

    return history
# train_modle(*) end


if __name__ == "__main__":
    history = train_model(max_epoch=100)
