import torch
import partialtorch


# noinspection DuplicatedCode
def test_partial_conv_module():
    x = torch.rand(1, 1, 3, 3)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :, :2, :2] = 0

    px = partialtorch.masked_pair(x, mask)
    px.fill_masked_(0)
    print('>>> x')
    print(x)
    print(mask)
    print()

    conv = partialtorch.nn.PartialConv2d(1, 1, kernel_size=(2, 2))

    # non-scaled op
    pout = partialtorch.partial_conv2d(px, conv.weight, None,
                                       conv.stride, conv.padding, conv.dilation, conv.groups, scaled=False)
    print('partial_conv2d[scaled=False]')
    print(pout.to_tensor(0))
    print()

    # scaled op
    pout = partialtorch.partial_conv2d(px, conv.weight, None,
                                       conv.stride, conv.padding, conv.dilation, conv.groups, scaled=True)
    print('partial_conv2d[scaled=True]')
    print(pout.to_tensor(0))
    print()

    # conv forward
    conv = torch.jit.script(conv)
    print(conv)
    pout = conv(px)
    print(pout.to_tensor(0))


# noinspection DuplicatedCode
def test_partial_conv_transpose_module():
    torch.manual_seed(1)
    x = torch.rand(1, 1, 3, 3)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :, :2, :2] = 0

    px = partialtorch.masked_pair(x, mask)
    px.fill_masked_(0)
    print(x)
    print(mask)
    print()

    conv = partialtorch.nn.PartialConvTranspose2d(1, 1, kernel_size=(2, 2))

    # non scaled op
    pout = partialtorch.partial_conv_transpose2d(px, conv.weight, None,
                                                 conv.stride, conv.padding, conv.output_padding,
                                                 conv.groups, conv.dilation, scaled=False)
    print('partial_conv_transpose2d[scaled=False]')
    print(pout.to_tensor(0))
    print()

    # scaled op
    pout = partialtorch.partial_conv_transpose2d(px, conv.weight, None,
                                                 conv.stride, conv.padding, conv.output_padding,
                                                 conv.groups, conv.dilation, scaled=True)
    print('partial_conv_transpose2d[scaled=True]')
    print(pout.to_tensor(0))
    print()

    # conv forward
    conv = torch.jit.script(conv)
    print(conv)
    pout = conv(px)
    print(pout.to_tensor(0))


if __name__ == '__main__':
    test_partial_conv_module()
    test_partial_conv_transpose_module()
