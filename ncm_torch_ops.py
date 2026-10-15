import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode


def _arguments(args, kwargs, names, defaults):
    if len(args) > len(names) or kwargs.keys() - set(names):
        return None
    if any(name in kwargs for name in names[:len(args)]):
        return None
    values = dict(zip(names, defaults))
    values.update(zip(names, args))
    values.update(kwargs)
    return values


def _single(value, expected):
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    return type(value) is int and value == expected


class NCMMPSSafeOps(TorchFunctionMode):
    def __init__(self, devices=("mps",)):
        self.devices = tuple(devices)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.interpolate:
            values = _arguments(
                args, kwargs,
                ("input", "size", "scale_factor", "mode", "align_corners",
                 "recompute_scale_factor", "antialias"),
                (None, None, None, "nearest", None, None, False),
            )
            if values is not None:
                x, size = values["input"], values["size"]
                if isinstance(size, (tuple, list)) and len(size) == 1:
                    size = size[0]
                if (isinstance(x, torch.Tensor) and x.ndim == 3
                        and x.device.type in self.devices
                        and all(dimension > 0 for dimension in x.shape)
                        and values["mode"] == "area"
                        and type(size) is int and size > 0
                        and values["scale_factor"] is None
                        and values["align_corners"] is None
                        and values["recompute_scale_factor"] in (None, False)
                        and values["antialias"] is False):
                    length = x.shape[-1]
                    return torch.stack([
                        x[..., index * length // size:
                          ((index + 1) * length + size - 1) // size].mean(dim=-1)
                        for index in range(size)
                    ], dim=-1)
        elif func in (F.conv1d, F.conv_transpose1d):
            transpose = func is F.conv_transpose1d
            names = (("input", "weight", "bias", "stride", "padding",
                      "output_padding", "groups", "dilation") if transpose else
                     ("input", "weight", "bias", "stride", "padding", "dilation", "groups"))
            defaults = ((None, None, None, 1, 0, 0, 1, 1) if transpose else
                        (None, None, None, 1, 0, 1, 1))
            values = _arguments(args, kwargs, names, defaults)
            if values is not None:
                x, weight, bias = values["input"], values["weight"], values["bias"]
                if (isinstance(x, torch.Tensor) and x.ndim == 3
                        and x.device.type in self.devices
                        and x.shape[1:] == (256, 1 if transpose else 89)
                        and isinstance(weight, torch.Tensor)
                        and weight.shape == (256, 256, 89)
                        and weight.device == x.device
                        and (bias is None or isinstance(bias, torch.Tensor)
                             and bias.shape == (256,) and bias.device == x.device)
                        and _single(values["stride"], 89)
                        and _single(values["padding"], 0)
                        and _single(values["dilation"], 1)
                        and type(values["groups"]) is int and values["groups"] == 1
                        and (not transpose or _single(values["output_padding"], 0))):
                    if transpose:
                        output = F.linear(x[..., 0], weight.flatten(1).t())
                        output = output.reshape(x.shape[0], 256, 89)
                        return output if bias is None else output + bias[None, :, None]
                    return F.linear(x.flatten(1), weight.flatten(1), bias).unsqueeze(-1)
        return func(*args, **kwargs)
