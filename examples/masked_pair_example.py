import torch

import partialtorch


def test_creation():
    x = torch.rand(4, 4)
    x[0] += 10
    mask = torch.bernoulli(torch.full(x.shape, 0.6)).bool()

    p = partialtorch.MaskedPair(x, mask)
    # p = partialtorch.masked_pair(x, mask)
    # p = partialtorch.masked_pair([x])
    print(p)
    print('data', p.data)
    print('mask', p.mask)
    print()

    p.mask = torch.bernoulli(torch.full(x.shape, 0.5)).bool()
    print(p)


def test_unpacking():
    x = torch.rand(4, 4)
    px = partialtorch.rand_mask_(x, 0.5)

    x, x_mask = px.members
    px.members = x, None
    print('data', x)
    print('mask', x_mask)


def test_unary():
    x = torch.rand(4, 4)
    px = partialtorch.rand_mask_(x, 0.6)

    print('>>> x', px, sep='\n')
    print()

    pout = partialtorch._ops.dropout1d_(px, 0.5, True)
    print(px)


def test_binary():
    x = torch.rand(4, 4)
    px = partialtorch.rand_mask_(x, 0.6)

    y = torch.rand(4, 4)
    py = partialtorch.rand_mask_(y, 0.6)

    print('>>> x', px, sep='\n')
    print('>>> y', py, sep='\n')
    print()

    pout = partialtorch._ops.add(px, py, alpha=1)
    print(pout)

    pout = partialtorch._ops.partial_add(px, py)
    print(pout)

    pout = partialtorch._ops.partial_add_(px, py, scaled=True)
    print(pout)


def test_partial_binary():
    x = torch.rand(4, 3, requires_grad=True)
    px = partialtorch.rand_mask_(x, 0.6)

    y = torch.rand(3, 2, requires_grad=True)
    py = partialtorch.rand_mask_(y, 0.6)

    print('>>> x', px, sep='\n')
    print('>>> y', py, sep='\n')

    pout = partialtorch._ops.partial_mm(px, py)
    print(pout)
    pout = partialtorch._ops.partial_mm(px, py, scaled=True)
    print(pout)
    print(torch.mm(px.mask.long(), py.mask.long()))


def test_partial_ternary():
    x = torch.rand(3, 3, requires_grad=True)
    px = partialtorch.rand_mask_(x, 0.5)
    y = torch.rand(3, 3, requires_grad=True)
    py = partialtorch.rand_mask_(y, 0.5)
    z = torch.rand(3, 3, requires_grad=True)
    pz = partialtorch.rand_mask_(z, 0.5)

    print('>>> x', px, sep='\n')
    print('>>> y', py, sep='\n')
    print('>>> z', pz, sep='\n')

    pout = partialtorch._ops.partial_addmm(px, py, pz)
    print(pout)
    print(torch.ops.aten.addmm(x * px.mask, y * py.mask, z * pz.mask))


def test_reduction():
    x = torch.rand(4, 4)
    px = partialtorch.rand_mask_(x, 0.2)
    print('>>> x', px, sep='\n')
    print()

    pout = partialtorch._ops.sum(px, (0,))
    assert torch.equal(pout.data,
                       partialtorch._ops.fill_masked(px, 0).data.sum((0,)))

    pout = partialtorch._ops.cumsum(px, 1)
    assert torch.equal(pout.data,
                       partialtorch._ops.fill_masked(px, 0).data.cumsum(1))

    p = -3
    pout = partialtorch._ops.norm(px, p=p)
    assert torch.equal(pout.data, x[px.mask].norm(p=p))

    pout = partialtorch._ops.std(px)
    assert torch.equal(pout.data, torch.masked._ops.std(px.data, mask=px.mask))

    pout = partialtorch._ops.softmin(px, dim=1)
    assert torch.equal(torch.nan_to_num(pout.data, 0),
                       torch.nan_to_num(torch.nn.functional.softmin(
                           partialtorch._ops.fill_masked(px, torch.inf).data, dim=1), 0))


def test_scaled_reduction():
    x = torch.rand(4, 4)
    px = partialtorch.rand_mask_(x, 0.2)
    print('>>> x', px, sep='\n')
    print()

    pout = partialtorch._ops.sum(px, (0,))
    print(pout)

    pout = partialtorch._ops.sum(px, (0,), scaled=True)
    print(pout)


def test_passthrough():
    x = torch.rand(4, 4)
    px = partialtorch.rand_mask_(x, 0.2)

    y = torch.rand(4, 4)
    py = partialtorch.rand_mask_(y, 0.2)

    print('>>> x', px, sep='\n')
    print('>>> y', py, sep='\n')
    print()

    pouts = partialtorch._ops.chunk(px, 2, 1)
    print(pouts[0].shape, pouts[1].shape)


if __name__ == '__main__':
    test_creation()
    # test_unpacking()
    # test_unary()
    # test_binary()
    # test_partial_binary()
    # test_partial_ternary()
    # test_reduction()
    # test_scaled_reduction()
    # test_passthrough()
