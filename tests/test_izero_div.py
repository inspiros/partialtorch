import torch
from tqdm import trange

from partialtorch.ops.izero_div import _izero_div, _izero_ldiv, _izero_ldiv_
from utils.time_meter import TimeMeter


def test_izero_div(dtype=torch.float64, device='cuda'):
    torch.manual_seed(1)
    x = torch.rand(4, 4, dtype=dtype, device=device, requires_grad=True)
    y = torch.rand(4, 4, dtype=dtype, device=device, requires_grad=True)

    grad_correct = torch.autograd.gradcheck(_izero_div, (x, y))
    print('grad_correct[izero_div]:', grad_correct)

    grad_correct = torch.autograd.gradcheck(_izero_ldiv, (x, y))
    print('grad_correct[izero_ldiv]:', grad_correct)

    x.data[:1, :1] = 0
    y.data[-1:, -1:] = 0

    print('x:', x, sep='\n')
    print('y:', y, sep='\n')
    print()

    out = _izero_div(x, y)
    native_out = torch.div(x, y)
    print(out - native_out)

    x.requires_grad_(False)
    native_out = torch.div(3, x)
    out = _izero_ldiv_(x, 3)
    print(x - native_out)

    x = torch.rand(128, 512, dtype=dtype, device=device, requires_grad=True)
    y = torch.rand(128, 512, dtype=dtype, device=device, requires_grad=True)
    mask = torch.bernoulli(torch.full_like(x, 0.8))
    y.data.mul_(mask)

    eps = torch.finfo(y.dtype).eps
    print('outputs match:', torch.allclose(
        _izero_div(x, y),
        x.div(y + eps).mul_(mask)
    ))

    time_meter = TimeMeter()
    native_time_meter = TimeMeter()

    # warmup
    out = _izero_div(x, y)
    grad = torch.ones_like(out)

    for iter_id in (pbar := trange(2000)):
        with time_meter:
            out = _izero_div(x, y)
            # out.backward(grad)
        with native_time_meter:
            out = torch.ops.aten.mul_(torch.ops.aten.div(x, torch.ops.aten.add(y, eps)), mask)
            # out.backward(grad)
        pbar.set_description(f'[Iter {iter_id + 1}] op_fps={time_meter.fps:.04f}, '
                             f'native_op_fps={native_time_meter.fps:.04f}')


if __name__ == '__main__':
    test_izero_div()
