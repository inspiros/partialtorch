import torch
import partialtorch


# noinspection DuplicatedCode
def test_masked_pool_module():
    x = torch.rand(1, 1, 4, 4)
    px = partialtorch.rand_mask_(x, 0.4, mask_value=torch.inf)
    print('>>> x')
    print(px)
    print()

    pool = partialtorch.nn.MaxPool2d(kernel_size=(2, 2))

    # op
    pout, indices = partialtorch.max_pool2d_with_indices(
        px, pool.kernel_size, pool.stride, pool.padding, pool.dilation, pool.ceil_mode)
    print('max_pool2d')
    print(pout.to_tensor(0))
    print(indices)
    print()

    # pool forward
    pool = torch.jit.script(pool)
    print(pool)
    pout = pool(px)
    print(pout.to_tensor(0))


if __name__ == '__main__':
    test_masked_pool_module()
