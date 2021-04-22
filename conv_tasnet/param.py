class ConvTasNetParam:

    __slots__ = 'causal', 'That', 'C', 'L', 'N', 'B', 'Sc', 'H', 'P', 'X', 'R', 'overlap'

    def __init__(self,
                 causal: bool = False,
                 That: int = 128,
                 C: int = 4,
                 L: int = 16,
                 N: int = 512,
                 B: int = 128,
                 Sc: int = 128,
                 H: int = 512,
                 P: int = 3,
                 X: int = 8,
                 R: int = 3,
                 overlap: int = 8):

        if overlap * 2 > L:
            raise ValueError('`overlap` cannot be greater than half of `L`!')

        self.causal = causal
        self.That = That
        self.C = C
        self.L = L
        self.N = N
        self.B = B
        self.Sc = Sc
        self.H = H
        self.P = P
        self.X = X
        self.R = R
        self.overlap = overlap

    def get_config(self) -> dict:
        return {'causal': self.causal,
                'That': self.That,
                'C': self.C,
                'L': self.L,
                'N': self.N,
                'B': self.B,
                'Sc': self.Sc,
                'H': self.H,
                'P': self.P,
                'X': self.X,
                'R': self.R,
                'overlap': self.overlap}

    def save(self, path: str):
        with open(path, 'w', encoding='utf8') as f:
            f.write('\n'.join(f"{k}={v}" for k, v
                    in self.get_config().items()))

    @staticmethod
    def load(path: str):
        def convert_int(value):
            try:
                return int(value)
            except:
                pass
            return value

        def convert_bool(value):
            if value == 'True':
                return True
            elif value == 'False':
                return False
            else:
                return value

        def convert_tup(tup):
            if tup[0] == 'causal':
                return (tup[0], convert_bool(tup[1]))
            else:
                return (tup[0], convert_int(tup[1]))

        with open(path, 'r', encoding='utf8') as f:
            d = dict(convert_tup(line.strip().split('='))
                     for line in f.readlines())
            return ConvTasNetParam(**d)

    def __str__(self) -> str:
        return f'Conv-TasNet Hyperparameters: {str(self.get_config())}'
