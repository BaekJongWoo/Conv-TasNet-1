class ConvTasNetParam():
    """Conv-TasNet Hyperparameters

    Attributes:
        K (int): Number of samples
        C (int): Number of sources
        L (int): Length of the filters (in samples)
        N (int): Number of filters in autoencoder
        B (int): Number of channels in bottleneck and the residual paths' 1x1-conv blocks
        S (int): Number of channels in skip-connection paths' 1x1-conv blocks
        H (int): Number of channels in convolutional blocks
        P (int): Kernel size in convolutional blocks
        X (int): Number of convolutional blocks in each repeat
        R (int): Number of repeats
        causal (bool): Causality of model
        gating (bool): Gating mechanism flag
        w_activation (str): Activation function for conv1d_U in the ConvTasNetEncoder
        m_activation (str): Activation function for estimated_masks in the ConvTasNetSeparator
        eps (float): Small constant for numerical stability  
    """

    def __init__(self,
                 K: int = 50, C: int = 3,
                 L: int = 16, N: int = 512,
                 B: int = 128, S: int = 128,
                 H: int = 512, P: int = 3,
                 X: int = 8, R: int = 3,
                 causal: bool = True,
                 gating: bool = True,
                 w_activation: str = "linear",
                 m_activation: str = "sigmoid",
                 eps: float = 1e-8):
        self.K, self.C = K, C
        self.L, self.N = L, N
        self.B, self.S = B, S
        self.H, self.P = H, P
        self.X, self.R = X, R
        self.causal = causal
        self.gating = gating
        self.w_activation = w_activation
        self.m_activation = m_activation
        self.eps = eps

    def get_config(self) -> dict:
        return {"K": self.K, "C": self.C,
                "L": self.L, "N": self.N,
                "B": self.B, "S": self.S,
                "H": self.H, "P": self.P,
                "X": self.X, "R": self.X,
                "Causal": self.causal,
                "gating": self.gating,
                "weight_activation": self.w_activation,
                "mask_activation": self.m_activation,
                "eps": self.eps}
# ConvTasNetParam end
