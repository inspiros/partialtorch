from partialtorch.types import Number, MaskedPair, _MaskedPairOrTensor


def addcmul(self: _MaskedPairOrTensor,
            tensor1: _MaskedPairOrTensor,
            tensor2: _MaskedPairOrTensor,
            value: Number = 1) -> MaskedPair: ...


def addcdiv(self: _MaskedPairOrTensor,
            tensor1: _MaskedPairOrTensor,
            tensor2: _MaskedPairOrTensor,
            value: Number = 1) -> MaskedPair: ...


# -----------------------
# partial ternary
# -----------------------
def partial_addmm(self: _MaskedPairOrTensor,
                  mat1: _MaskedPairOrTensor,
                  mat2: _MaskedPairOrTensor,
                  beta: Number = 1,
                  alpha: Number = 1) -> MaskedPair: ...


def partial_addbmm(self: _MaskedPairOrTensor,
                   batch1: _MaskedPairOrTensor,
                   batch2: _MaskedPairOrTensor,
                   beta: Number = 1,
                   alpha: Number = 1) -> MaskedPair: ...


def partial_baddbmm(self: _MaskedPairOrTensor,
                    batch1: _MaskedPairOrTensor,
                    batch2: _MaskedPairOrTensor,
                    beta: Number = 1,
                    alpha: Number = 1) -> MaskedPair: ...


def partial_addmv(self: _MaskedPairOrTensor,
                  mat: _MaskedPairOrTensor,
                  vec: _MaskedPairOrTensor,
                  beta: Number = 1,
                  alpha: Number = 1) -> MaskedPair: ...


def partial_addr(self: _MaskedPairOrTensor,
                 vec1: _MaskedPairOrTensor,
                 vec2: _MaskedPairOrTensor,
                 beta: Number = 1,
                 alpha: Number = 1) -> MaskedPair: ...


def partial_addcmul(self: _MaskedPairOrTensor,
                    tensor1: _MaskedPairOrTensor,
                    tensor2: _MaskedPairOrTensor,
                    alpha: Number = 1) -> MaskedPair: ...


def partial_addcdiv(self: _MaskedPairOrTensor,
                    tensor1: _MaskedPairOrTensor,
                    tensor2: _MaskedPairOrTensor,
                    alpha: Number = 1) -> MaskedPair: ...
