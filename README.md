![logo](https://raw.githubusercontent.com/inspiros/partialtorch/master/resources/logo.png) PartialTorch ![Build Wheels Status](https://img.shields.io/github/actions/workflow/status/inspiros/partialtorch/build_wheels.yml) ![License](https://img.shields.io/github/license/inspiros/partialtorch)
=============

**PartialTorch** is a thin C++ wrapper of **PyTorch**'s operators to support masked and partial semantics.

## Main Features

### Masked Pair

We use a custom C++ extension class called `partialtorch.MaskedPair` to store ``data`` and ``mask`` (an optional
``Tensor`` of the same shape as ``data``, containing ``0/1`` values indicating the availability of the corresponding
element in ``data``).

The advantages of `MaskedPair` is that it is statically-typed but unpackable like `namedtuple`,
and more importantly, it is accepted by `torch.jit.script` functions as argument or return type.
This container is a temporary substitution for `torch.masked.MaskedTensor` and may change in the future.

This table compares the two in some aspects:

|                                     |                             ``torch.masked.MaskedTensor``                              |                      ``partialtorch.MaskedPair``                       |
|:------------------------------------|:--------------------------------------------------------------------------------------:|:----------------------------------------------------------------------:|
| **Backend**                         |                                         Python                                         |                                  C++                                   |
| **Nature**                          |          Is a subclass of ``Tensor`` with ``mask`` as an additional attribute          |                Is a container of ``data`` and ``mask``                 |
| **Supported layouts**               |                                   Strided and Sparse                                   |                             Only Strided️                              |
| **Mask types**                      |                                  ``torch.BoolTensor``                                  |       ``Optional[torch.BoolTensor]`` (may support other dtypes)        |
| **Ops Coverage**                    | Listed [here](https://pytorch.org/docs/stable/masked.html) (with lots of restrictions) |  All masked ops that ``torch.masked.MaskedTensor`` supports and more   |
| **``torch.jit.script``-able**       |            Yes✔️ (Python ops seem not to be jit compiled but encapsulated)             |                                 Yes✔️                                  |
| **Supports ``Tensor``'s methods**   |                                         Yes✔️                                          |                             Only a few[^1]                             |
| **Supports ``__torch_function__``** |                                         Yes✔️                                          |                                No❌[^1]                                 |
| **Performance**                     |           Slow and sometimes buggy (e.g. try calling ``.backward`` 3 times)            | Faster, not prone to bugs related to ``autograd`` as it is a container |

[^1]: We blame ``torch`` 😅

More details about the differences will be discussed below.

### Masked Operators

<p align="center">
    <img src="https://raw.githubusercontent.com/inspiros/partialtorch/master/resources/torch_masked_binary.png" width="600">
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/inspiros/partialtorch/master/resources/masked_binary.png" width="600">
</p>

**Masked operators** are the same things that can be found in ``torch.masked``
package (_which is, unfortunately, still in prototype stage_).

Our semantic differs from ``torch.masked`` for non-unary operators.

- ``torch.masked``: Requires operands to share identical mask
  (check this [link](https://pytorch.org/docs/stable/masked.html)), which is not always the case when we have to deal
  with missing data.
- ``partialtorch``: Allows operands to have different masks, the output mask is the result of a _bitwise all_ function
  of input masks' values.

### Partial Operators

<p align="center">
    <img src="https://raw.githubusercontent.com/inspiros/partialtorch/master/resources/partial_binary.png" width="600">
</p>

Similar to masked operators, **partial operators** allow non-uniform masks but instead of using _bitwise all_
to compute output mask, they use _bitwise any_.
That means output at any position with at least one present operand is NOT considered missing.

In details, before fowarding to the regular ``torch`` native operators, the masked positions of each operand are filled
with an _identity value_.
The identity value is defined as the initial value that has the property ``op(op_identity, value) = value``.
For example, the identity value of element-wise addition is ``0``.

<p align="center">
    <img src="https://raw.githubusercontent.com/inspiros/partialtorch/master/resources/regular_binary.png" width="600">
</p>

All partial operators have a prefix ``partial_`` prepended to their name (e.g. ``partialtorch.partial_add``),
while masked operators inherit their native ops' names.
Reduction operators are excluded from this rule as they can be considered unary partial, and some of them
are already available in ``torch.masked``.

#### Scaled Partial Operators

<p align="center">
    <img src="https://raw.githubusercontent.com/inspiros/partialtorch/master/resources/scaled_binary.png" width="800">
</p>

Some partial operators that involves addition/substraction are extended to have _rescaling semantic_.
We call them **scaled partial operators**.
In essence, they rescale the output by the ratio of present operands in the computation of the output.
The idea is similar to ``torch.dropout`` rescaling by $\frac{1}{1-p}$,
or more precisely the way [**Partial Convolution**](https://arxiv.org/abs/1804.07723) works.

Programatically, all scaled partial operators share the same signature with their non-scaled counterparts,
and are dispatched to when adding a keyword-only argument ``scaled = True``:

```python
pout = partialtorch.partial_add(pa, pb, scaled=True)
```

### Torch Ops Coverage

We found out that the workload is behemoth for a group of one person, and involves manually reimplementing all
native functors under the ``at::_ops`` namespace (guess how many there are).
Therefore, we try to cover as many primitive operators as possible, as well as a few other operators relevant to our
work.
The full list of all registered signatures can be found in this [file](resources/partialtorch_ops.yaml).

If you want any operator to be added, please contact me.
But if they fall into one of the following categories, the porting may take long or will not happen:

- Ops that do not have a meaningful masked semantic (e.g. ``torch.det``).
- Ops that cannot be implemented easily by calling native ops and requires writing custom kernels (e.g. ``torch.mode``).
- Ops that accept output as an input a.k.a. _out_ ops (e.g.
  ``aten::mul.out(self: Tensor, other: Tensor, *, out: Tensor(a!)) -> Tensor(a!)``).
- Ops for named tensors (e.g. with argument ``dim: Dimname`` or ``dims: DimnameList`` in schema).
- Ops for tensors with unsuported properties (e.g. sparse, quantized layouts).
- Ops with any input/return type that do not have ``pybind11`` type conversions predefined by ``torch``'s C++ backend.

Also, everyone is welcome to contribute.

## Requirements

- ``torch>=2.1.0`` _(this version of **PyTorch** brought a number of changes that are not backward compatible)_

## Installation

#### From TestPyPI

[partialtorch](https://test.pypi.org/project/partialtorch/) has wheels hosted at **TestPyPI**
(it is not likely to reach a stable state anytime soon):

```bash
pip install -i https://test.pypi.org/simple/ partialtorch
```

The Linux and Windows wheels are built with **Cuda 12.1**.
If you cannot find a wheel for your Arch/Python/Cuda, or there is any problem with library linking when importing,
proceed to [instructions to build from source](#from-source).

|                  |             Linux/Windows             |     MacOS      |
|------------------|:-------------------------------------:|:--------------:|
| Python version:  |               3.8-3.11                |    3.8-3.11    |
| PyTorch version: |            `torch==2.1.0`             | `torch==2.1.0` |
| Cuda version:    |                 12.1                  |       -        |
| GPU CCs:         | `5.0,6.0,6.1,7.0,7.5,8.0,8.6,9.0+PTX` |       -        |

#### From Source

For installing from source, you need a C++17 compiler (`gcc`/`msvc`) and a Cuda compiler (`nvcc`) installed.
Then, clone this repo and execute:

```bash
pip install .
```

## Usage

### Initializing a ``MaskedPair``

While ``MaskedPair`` is almost as simple as a ``namedtuple``, there are also a few supporting creation ops:

```python
import torch, partialtorch

x = torch.rand(3, 3)
x_mask = torch.bernoulli(torch.full_like(x, 0.5)).bool()  # x_mask must have dtype torch.bool

px = partialtorch.masked_pair(x, x_mask)  # with 2 inputs data and mask
px = partialtorch.masked_pair(x)  # with data only (mask = None)
px = partialtorch.masked_pair(x, None)  # explicitly define mask = None
px = partialtorch.masked_pair(x, True)  # explicitly define mask = True (equivalent to None)
px = partialtorch.masked_pair((x, x_mask))  # from tuple

# this new random function conveniently does the work of the above steps
px = partialtorch.rand_mask(x, 0.5)
```

Note that ``MaskedPair`` is not a subclass of ``Tensor`` like ``MaskedTensor``,
so we only support a very limited number of methods.
This is mostly because of the current limitations of C++ backend for custom classes[^1] such as:

- Unable to overload methods with the same name
- Unable to define custom type conversions from Python type (``Tensor``) or to custom Python type
  (to be able to define custom methods such as ``__str__`` of ``Tensor`` does for example)
- Unable to define ``__torch_function__``

In the meantime, please consider ``MaskedPair`` purely a fast container and use
``partialtorch.op(pair, ...)`` instead of ``pair.op(...)`` if not available.

**Note:** You cannot index ``MaskedPair`` with ``pair[..., 1:-1]`` as they acts like tuple of 2 elements when indexed.

### Operations

All registered ops can be accessed like any torch's custom C++ operator by calling ``torch.ops.partialtorch.[op_name]``
(the same way we call native ATen function ``torch.ops.aten.[op_name]``).
Their overloaded versions that accept ``Tensor`` are also registered for convenience
(but return type is always converted to ``MaskedPair``).

```python
import torch, partialtorch

x = torch.rand(5, 5)
px = partialtorch.rand_mask(x, 0.5)

# traditional
pout = torch.ops.partialtorch.sum(px, 0, keepdim=True)
# all exposed ops should be aliased inside partialtorch.ops
pout = partialtorch.ops.sum(px, 0, keepdim=True)
```

Furthermore, we inherit the naming convention of for inplace ops - appending a trailing ``_`` after their names
(``partialtorch.relu`` and ``partialtorch.relu_``).
They modify both data and mask of the first operand inplacely.

The usage is kept as close to the corresponding ``Tensor`` ops as possible.
Hence, further explaination is redundant.
A few examples can be found in [examples](examples) folder.

### Neural Network Layers

Only some layers are implemented in `partialtorch.nn` sub package.

- `PartialConvNd`, `PartialConvTransposeNd`: [examples/partial_conv_example.py](examples/partial_conv_example.py)
- ``MaskedMaxPoolNd``, ``MaskedFractionalMaxPoolNd`` [examples/masked_max_pool_example.py](examples/masked_max_pool_example.py)
- `MaskedBatchNormNd`: [examples/masked_batchnorm_example.py](examples/masked_batchnorm_example.py)
- _More to be added_

## Citation

This code is part of another project of us. Citation will be added in the future.

## Acknowledgements

Part of the codebase is modified from the following repositories:

- https://github.com/pytorch/pytorch
- https://github.com/NVIDIA/partialconv

## License

The code is released under the MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.
