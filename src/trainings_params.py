class TrainingParams:
    def __init__(self, exp_name: str) -> None:
        self.exp_name: str = exp_name
        self.dim_latent_space: int = 2
        self.init_values: list = [1.8, 4.25, 0.05, 1, 1, 1]
        self.lamda_train: float = 0.95
        self.max_epochs: int = 200
        self.lr_shed_patience: int = 4
        self.lr: float = 0.05
        self.compression: float = 0.1
        self.params_dl = {
            "batch_size": 160,
            "shuffle": True,
            "num_workers": 0,
            "drop_last": False,
            "pin_memory": False,
        }


class Demo3D(TrainingParams):
    def __init__(self) -> None:
        super().__init__("3d_demo")

        self.image_size_polar: list[int] | None = None
        self.image_size_cartesian: list[int] | None = None
