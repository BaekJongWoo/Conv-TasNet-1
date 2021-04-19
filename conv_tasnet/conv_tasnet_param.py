class ConvTasNetParam():
    """Hyperparameter Container.

    Attributes:
        K (int): Number of input segments
        C (int): Number of sources
        L (int): Length of the filters in samples
        N (int): Number of filters in autoencoder
        B (int): Number of channels in bottleneck and the residual paths' 1x1-conv blocks
        S (int): Number of channels in skip-connection paths' 1x1-conv blocks
        H (int): Number of channels in convolutional blocks
        P (int): Kernel size in convolutional blocks
        X (int): Number of convolutional blocks in each repeat
        R (int): Number of repeats
        causal (bool): Causality of model
        gating (bool): Option for gating mechanism
        use_bias (bool): Option for 1-D convolution bias
        eps (float): Small constant for numerical stability  
    """

    def __init__(self,
                 K: int = 128,
                 C: int = 4,
                 L: int = 16,
                 N: int = 512,
                 B: int = 128,
                 S: int = 128,
                 H: int = 512,
                 P: int = 3,
                 X: int = 8,
                 R: int = 3,
                 causal: bool = True,
                 gating: bool = False,
                 use_bias: bool = False,
                 eps: float = 1e-8):
        self.K = K
        self.C = C
        self.L = L
        self.N = N
        self.B = B
        self.S = S
        self.H = H
        self.P = P
        self.X = X
        self.R = R
        self.causal = causal
        self.gating = gating
        self.use_bias = use_bias
        self.eps = eps

    def get_config(self) -> dict:
        return {"K": self.K,
                "C": self.C,
                "L": self.L,
                "N": self.N,
                "B": self.B,
                "S": self.S,
                "H": self.H,
                "P": self.P,
                "X": self.X,
                "R": self.R,
                "causal": self.causal,
                "gating": self.gating,
                "use_bias": self.use_bias,
                "eps": self.eps}

    def __str__(self) -> str:
        return str(self.get_config())
# ConvTasNetParam end
