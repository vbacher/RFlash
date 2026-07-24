import torch


def Normal_initialization(shape_cost_vol: tuple, constant_init_vals: list[float]) -> torch.Tensor:
    """Returns parameter maps with Hughes initialization

    Args:
        shape_cost_vol (tuple): Shape of cost volume
        constant_init_vals (list[float]): Constant values used for initialiation

    Returns:
        torch.Tensor: Parameter maps
    """
    init_list = []

    # init attenuation
    init_list.append(
        torch.empty(shape_cost_vol).normal_(constant_init_vals[0], constant_init_vals[0] * 0.0005)[:, :, :, None]
    )  # 0.0005

    # init scatter
    init_list.append(
        torch.empty(shape_cost_vol).normal_(constant_init_vals[1], constant_init_vals[1] * 0.0005)[:, :, :, None]
    )  # 0.0005

    # combine
    init_vol = torch.cat(tuple(init_list), dim=3)

    return init_vol
