from typing import Sequence

import numpy as np
import torch

import partialtorch


def test_masked_batchnorm(masked=True, p=0.5, feat_dims=(5,), feat_mean=10., feat_std=2.):
    if not isinstance(feat_dims, Sequence):
        feat_dims = (feat_dims,)
    else:
        feat_dims = tuple(feat_dims)

    print(f'true_mean={torch.full(feat_dims, feat_mean)}')
    print(f'true_std={torch.full(feat_dims, feat_std)}')

    ndim = 1 if len(feat_dims) in [1, 2] else len(feat_dims) - 1
    if masked:
        bn_class = getattr(partialtorch.nn, f'MaskedBatchNorm{ndim}d')
    else:
        bn_class = getattr(torch.nn, f'BatchNorm{ndim}d')
    bn = bn_class(feat_dims[0])
    print(f'\n[Using {bn.__class__.__name__}]')
    bn = torch.jit.script(bn)
    print(bn)

    torch.manual_seed(1)
    np.random.seed(1)

    def get_random_feats(batch_size=1):
        return torch.from_numpy(np.random.normal(
            feat_mean, feat_std, (batch_size, *feat_dims)).astype(np.float32))

    # training behavior
    print('Training')
    bn.train()
    for step in range(100):
        x = get_random_feats(10)
        px = partialtorch.rand_mask(x, p=p, mask_value=0)

        if masked:
            py = bn(px)
        else:
            y = bn(x)
    print(f'running_mean={bn.running_mean}')
    print(f'running_std={bn.running_var.sqrt()}')
    print()

    # eval behavior
    print('Testing')
    bn.eval()
    x = get_random_feats(256)
    px = partialtorch.rand_mask(x, p=p, mask_value=0)
    if masked:
        py = bn(px)
        print(f'test_mean={partialtorch.mean(py, dim=0).data}')
        print(f'test_std={partialtorch.std(py, dim=0).data}')
    else:
        y = bn(x)
        print(f'test_mean={y.mean(dim=0)}')
        print(f'test_std={y.std(dim=0)}')


if __name__ == '__main__':
    test_masked_batchnorm()
