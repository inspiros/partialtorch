from typing import Sequence, Union

import numpy as np
import partialtorch
import torch
from torch import Tensor, Generator


class FeaturesGenerator(torch.nn.Module):
    def __init__(self,
                 sizes: Union[int, Sequence[int]],
                 mean: float = 0.,
                 std: float = 1.,
                 generator: Generator = None,
                 dtype=None,
                 device=None):
        super().__init__()
        self.factory_kwargs = {'dtype': dtype, 'device': device}
        if not isinstance(sizes, Sequence):
            self.sizes = (sizes,)
        else:
            self.sizes = tuple(sizes)
        self.register_buffer('mean', torch.tensor(mean, **self.factory_kwargs))
        self.register_buffer('std', torch.tensor(std, **self.factory_kwargs))
        self.generator = generator

    def get(self, batch_size: int = 1) -> Tensor:
        shape = (batch_size, *self.sizes)
        return torch.normal(self.mean.expand(shape), self.std.expand(shape), generator=self.generator)


def test_batch_norm(masked=True, p=0.5, feat_dims=(3, 5,), feat_mean=10., feat_std=2.):
    feats_generator = FeaturesGenerator(sizes=feat_dims, mean=feat_mean, std=feat_std)
    print(f'true_mean={torch.full(feat_dims, feat_mean)}')
    print(f'true_std={torch.full(feat_dims, feat_std)}')

    ndim = 1 if len(feat_dims) in [1, 2] else len(feat_dims) - 1
    bn_class = getattr(partialtorch.nn if masked else torch.nn, f'BatchNorm{ndim}d')
    bn = bn_class(feat_dims[0])
    bn = torch.jit.script(bn)
    print(bn)

    # torch.manual_seed(1)
    # np.random.seed(1)

    # training behavior
    print('Training')
    bn.train()
    for step in range(100):
        x = feats_generator.get(1)
        if masked:
            px = partialtorch.rand_mask(x, p=p, mask_value=0)
            py = bn(px)
        else:
            y = bn(x)
    print(f'running_mean={bn.running_mean}')
    print(f'running_std={bn.running_var.sqrt()}')
    print()

    # eval behavior
    print('Testing')
    bn.eval()
    x = feats_generator.get(256)
    if masked:
        px = partialtorch.rand_mask(x, p=p, mask_value=0)
        py = bn(px)
        print(f'test_mean={partialtorch.mean(py, dim=0).data}')
        print(f'test_std={partialtorch.std(py, dim=0).data}')
    else:
        y = bn(x)
        print(f'test_mean={y.mean(dim=0)}')
        print(f'test_std={y.std(dim=0)}')


def test_layer_norm(masked=True, p=0.5, feat_dims=(3, 5,), feat_mean=10., feat_std=2.):
    feats_generator = FeaturesGenerator(sizes=feat_dims, mean=feat_mean, std=feat_std)
    print(f'true_mean={torch.full(feat_dims, feat_mean)}')
    print(f'true_std={torch.full(feat_dims, feat_std)}')

    ndim = 1 if len(feat_dims) in [1, 2] else len(feat_dims) - 1
    ln_class = getattr(partialtorch.nn if masked else torch.nn, f'LayerNorm')
    ln = ln_class(feat_dims)
    ln = torch.jit.script(ln)
    print(ln)

    # torch.manual_seed(1)
    # np.random.seed(1)

    # training behavior
    print('Training')
    ln.train()
    for step in range(100):
        x = feats_generator.get(1)
        if masked:
            px = partialtorch.rand_mask(x, p=p, mask_value=0)
            py = ln(px)
        else:
            y = ln(x)
    print()

    # eval behavior
    print('Testing')
    ln.eval()
    x = feats_generator.get(256)
    if masked:
        px = partialtorch.rand_mask(x, p=p, mask_value=0)
        py = ln(px)
        print(f'test_mean={partialtorch.mean(py, dim=tuple(range(1, x.ndim))).data}')
        print(f'test_std={partialtorch.std(py, dim=tuple(range(1, x.ndim)), unbiased=False).data}')
    else:
        y = ln(x)
        print(f'test_mean={y.mean(dim=tuple(range(1, x.ndim)))}')
        print(f'test_std={y.std(dim=tuple(range(1, x.ndim)), unbiased=False)}')


if __name__ == '__main__':
    # test_batch_norm()
    test_layer_norm()
