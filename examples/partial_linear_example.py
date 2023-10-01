import torch

import partialtorch


def py_partial_linear(input, weight, bias=None, *, scaled=True, eps=1e-8):
    output_data = torch.nn.functional.linear(input.data, weight)
    if input.mask is None:
        return partialtorch.masked_pair(output_data)
    if scaled:
        with torch.no_grad():
            mask_ratio = torch.nn.functional.linear(input.mask.to(weight.dtype),
                                                    torch.ones_like(weight)) / weight.size(1)
            mask_ratio = 1 / (mask_ratio + eps)
        output_data *= mask_ratio
    if bias is not None:
        output_data += bias.view(1, -1)
    return partialtorch.masked_pair(output_data)


def test_partial_linear_module():
    x = torch.rand(3, 10)
    px = partialtorch.rand_mask_(x, 0.5, mask_value=0)
    print('>>> x', px, sep='\n')
    print(x)
    print()

    linear = partialtorch.nn.PartialLinear(10, 3, scaled=True)

    # non-scaled op
    pout = partialtorch.partial_linear(px, linear.weight, None, scaled=False)
    print('partial_linear[scaled=False]')
    print(pout.to_tensor(0))
    print()

    # python non-scaled op
    pout = py_partial_linear(px, linear.weight, None, scaled=False)
    print('py_partial_linear[scaled=False]')
    print(pout.to_tensor(0))
    print()

    # scaled op
    pout = partialtorch.partial_linear(px, linear.weight, None, scaled=True)
    print('partial_linear[scaled=True]')
    print(pout.to_tensor(0))
    print()

    # python scaled op
    pout = py_partial_linear(px, linear.weight, None, scaled=True)
    print('py_partial_linear[scaled=True]')
    print(pout.to_tensor(0))
    print()

    # linear forward
    linear = torch.jit.script(linear)
    print(linear)
    pout = linear(px)
    print(pout.to_tensor(0))


if __name__ == '__main__':
    test_partial_linear_module()
