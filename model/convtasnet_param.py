class ConvTasNetParam():
    """Hyperparameters Container

    Attributes:
        T_hat (int): Number of samples
        C  (int): Number of sources
        N  (int): Number of filters in autoencoder
        L  (int): Length of the filters (in sample)
        B  (int): Number of channels in bottleneck and the residual paths' 1x1-conv blocks
        Sc (int): Number of channels in skip-connection paths' 1x1-conv blocks
        H  (int): Number of channels in convolutional blocks
        P  (int): Kernel size in convolutional blocks
        X  (int): Number of convolutional blocks in each repeat
        R  (int): Number of repeats
        causality (bool): Causality of the model
        gating (bool): Flag for gating mechanism
        eps (float): Small constant for numerical stability
    """

    __slots__ = ("T_hat", "C", "N", "L", "B", "Sc", "H", "P",
                 "X", "R", "causality", "gating", "eps")

    def __init__(self, T_hat: int = 50, C: int = 3, N: int = 512, L: int = 16, B: int = 128, Sc: int = 128, H: int = 512,
                 P: int = 3, X: int = 8, R: int = 3, causality: bool = True, gating: bool = False, eps: float = 1e-8):
        self.T_hat, self.C = T_hat, C
        self.N, self.L = N, L
        self.B, self.Sc = B, Sc
        self.H, self.P = H, P
        self.X, self.R = X, R
        self.causality = causality
        self.gating = gating
        self.eps = eps

    def get_config(self):
        return {"T_hat": self.T_hat, "C": self.C,
                "N": self.N, "L": self.L,
                "B": self.B, "Sc": self.Sc,
                "H": self.H, "P": self.P,
                "X": self.X, "R": self.R,
                "causality": self.causality,
                "gating": self.gating,
                "eps": self.eps}
# ConvTasNetParam end


def get_param(C: int = 3, causality: bool = True, gating: bool = False, eps: float = 1e-8) -> ConvTasNetParam:
    return ConvTasNetParam(C=C, causality=causality, gating=gating, eps=eps)
