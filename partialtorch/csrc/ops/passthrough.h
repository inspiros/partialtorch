#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

// ops declaration macros
// one to one
#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OP(NAME, SELF_T) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self);

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OP_WITH(NAME, SELF_T, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self, ARG1);

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OP_WITH2(NAME, SELF_T, ARG1, ARG2) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self, ARG1, ARG2);

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OP_WITH3(NAME, SELF_T, ARG1, ARG2, ARG3) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self, ARG1, ARG2, ARG3);

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OP_(NAME, SELF_T) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_(SELF_T self);

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OP__WITH(NAME, SELF_T, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_(SELF_T self, ARG1);

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OP__WITH2(NAME, SELF_T, ARG1, ARG2) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_(SELF_T self, ARG1, ARG2);

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OP__WITH3(NAME, SELF_T, ARG1, ARG2, ARG3) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_(SELF_T self, ARG1, ARG2, ARG3);

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>)

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OP_WITH(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1)

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OP_WITH2(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2)

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, ARG1, ARG2, ARG3) \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OP_WITH3(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2, ARG3)

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OP_(NAME, intrusive_ptr_arg_t<TensorMaskedPair>)

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OP__WITH(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1)

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OP__WITH2(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2)

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, ARG1, ARG2, ARG3) \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OP__WITH3(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2, ARG3)

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME)                  \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS(NAME)

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1)                  \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1)

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2)                  \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2)

#define PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, ARG1, ARG2, ARG3) \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, ARG1, ARG2, ARG3)                  \
PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, ARG1, ARG2, ARG3)

// one to many
#define PT_DECLARE_ONE2MANY_PASSTHROUGH_OP(NAME, SELF_T) \
PARTIALTORCH_API std::vector<c10::intrusive_ptr<TensorMaskedPair>> NAME(SELF_T self);

#define PT_DECLARE_ONE2MANY_PASSTHROUGH_OP_WITH(NAME, SELF_T, ARG1) \
PARTIALTORCH_API std::vector<c10::intrusive_ptr<TensorMaskedPair>> NAME(SELF_T self, ARG1);

#define PT_DECLARE_ONE2MANY_PASSTHROUGH_OP_WITH2(NAME, SELF_T, ARG1, ARG2) \
PARTIALTORCH_API std::vector<c10::intrusive_ptr<TensorMaskedPair>> NAME(SELF_T self, ARG1, ARG2);

#define PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_ONE2MANY_PASSTHROUGH_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>)

#define PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_ONE2MANY_PASSTHROUGH_OP_WITH(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1)

#define PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_ONE2MANY_PASSTHROUGH_OP_WITH2(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2)

// many to one
#define PT_DECLARE_MANY2ONE_PASSTHROUGH_OP(NAME, SELF_T) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self);

#define PT_DECLARE_MANY2ONE_PASSTHROUGH_OP_WITH(NAME, SELF_T, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self, ARG1);

#define PT_DECLARE_MANY2ONE_PASSTHROUGH_OP_WITH2(NAME, SELF_T, ARG1, ARG2) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self, ARG1, ARG2);

#define PT_DECLARE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_MANY2ONE_PASSTHROUGH_OP(NAME, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>)

#define PT_DECLARE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_MANY2ONE_PASSTHROUGH_OP_WITH(NAME, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>, ARG1)

#define PT_DECLARE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_MANY2ONE_PASSTHROUGH_OP_WITH2(NAME, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>, ARG1, ARG2)

// many to many
#define PT_DECLARE_MANY2MANY_PASSTHROUGH_OP(NAME, SELF_T) \
PARTIALTORCH_API std::vector<c10::intrusive_ptr<TensorMaskedPair>> NAME(SELF_T self);

#define PT_DECLARE_MANY2MANY_PASSTHROUGH_OP_WITH(NAME, SELF_T, ARG1) \
PARTIALTORCH_API std::vector<c10::intrusive_ptr<TensorMaskedPair>> NAME(SELF_T self, ARG1);

#define PT_DECLARE_MANY2MANY_PASSTHROUGH_OP_WITH2(NAME, SELF_T, ARG1, ARG2) \
PARTIALTORCH_API std::vector<c10::intrusive_ptr<TensorMaskedPair>> NAME(SELF_T self, ARG1, ARG2);

#define PT_DECLARE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_MANY2MANY_PASSTHROUGH_OP(NAME, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>)

#define PT_DECLARE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_MANY2MANY_PASSTHROUGH_OP_WITH(NAME, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>, ARG1)

#define PT_DECLARE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_MANY2MANY_PASSTHROUGH_OP_WITH2(NAME, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>, ARG1, ARG2)

namespace partialtorch {
    namespace ops {
        // to
        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> to(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                c10::optional<at::ScalarType> dtype = c10::nullopt,
                c10::optional<at::Layout> layout = c10::nullopt,
                c10::optional<at::Device> device = c10::nullopt,
                c10::optional<bool> pin_memory = c10::nullopt,
                bool non_blocking = false,
                bool copy = false,
                c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> to(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::Device device,
                at::ScalarType dtype,
                bool non_blocking = false,
                bool copy = false,
                c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> to(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::ScalarType dtype,
                bool non_blocking = false,
                bool copy = false,
                c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> to(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const_intrusive_ptr_arg_t<TensorMaskedPair> other,
                bool non_blocking = false,
                bool copy = false,
                c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> to(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const at::Tensor &other,
                bool non_blocking = false,
                bool copy = false,
                c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                cpu)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                cuda)

        // one to one
        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                clone, c10::optional<at::MemoryFormat> memory_format = c10::nullopt)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                contiguous, at::MemoryFormat memory_format = at::MemoryFormat::Contiguous)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                detach)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                detach_copy)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                atleast_1d)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                atleast_2d)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                atleast_3d)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                diag, int64_t diagonal = 0)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                diag_embed, int64_t offset = 0, int64_t dim1 = 0, int64_t dim2 = 1)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                diagflat, int64_t offset = 0)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                diagonal, int64_t offset = 0, int64_t dim1 = 0, int64_t dim2 = 1)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                narrow, int64_t dim, c10::SymInt start, c10::SymInt length)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                narrow, int64_t dim, const at::Tensor &start, c10::SymInt length)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                narrow_copy, int64_t dim, c10::SymInt start, c10::SymInt length)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                select, int64_t dim, c10::SymInt index)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                repeat, at::SymIntArrayRef repeats)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                repeat_interleave,
                c10::SymInt repeats,
                c10::optional<int64_t> dim,
                c10::optional<int64_t> output_size = c10::nullopt)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                repeat_interleave,
                const at::Tensor &repeats,
                c10::optional<int64_t> dim,
                c10::optional<int64_t> output_size = c10::nullopt)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                tile, at::IntArrayRef dims)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                flatten, int64_t start_dim = 0, int64_t end_dim = 0)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                unflatten, int64_t dim, at::IntArrayRef sizes)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                broadcast_to, c10::SymIntArrayRef size)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                expand, c10::SymIntArrayRef size, bool implicit = false)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                expand_as, const at::Tensor &other)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                expand_as, const_intrusive_ptr_arg_t < TensorMaskedPair > other)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                reshape, c10::SymIntArrayRef size)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                reshape_as, const at::Tensor &other)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                reshape_as, const_intrusive_ptr_arg_t < TensorMaskedPair > other)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                view, c10::SymIntArrayRef size)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                view, at::ScalarType dtype)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                view_as, const at::Tensor &other)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                view_as, const_intrusive_ptr_arg_t < TensorMaskedPair > other)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(squeeze)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                squeeze, int64_t dim)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                squeeze, at::IntArrayRef dim)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                unsqueeze, int64_t dim)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(matrix_H)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                moveaxis, at::IntArrayRef source, at::IntArrayRef destination)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                moveaxis, int64_t source, int64_t destination)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                movedim, at::IntArrayRef source, at::IntArrayRef destination)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                movedim, int64_t source, int64_t destination)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                swapaxes, int64_t axis0, int64_t axis1)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                swapdims, int64_t axis0, int64_t axis1)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(t)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                transpose, int64_t dim0, int64_t dim1)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                permute, at::IntArrayRef dims)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                permute_copy, at::IntArrayRef dims)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                take, const at::Tensor &index)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                take_along_dim, const at::Tensor &indices, c10::optional<int64_t> dim = c10::nullopt)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                gather, int64_t dim , const at::Tensor & index, bool sparse_grad = false)

        PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                unfold, int64_t dimension, int64_t size, int64_t step)

        // one to many
        PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                chunk, int64_t chunks, int64_t dim = 0)

        PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                split, c10::SymInt split_size, int64_t dim = 0)

        PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                split, c10::SymIntArrayRef split_size, int64_t dim = 0)

        PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                split_with_sizes, c10::SymIntArrayRef split_sizes, int64_t dim = 0)

        PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                split_copy, c10::SymInt split_size, int64_t dim = 0)

        PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                split_with_sizes_copy, c10::SymIntArrayRef split_sizes, int64_t dim = 0)

        PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                dsplit, int64_t sections)

        PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                dsplit, at::IntArrayRef indices)

        PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                hsplit, int64_t sections)

        PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                hsplit, at::IntArrayRef indices)

        PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                vsplit, int64_t sections)

        PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                vsplit, at::IntArrayRef indices)

        // many to one
        PT_DECLARE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                cat, int64_t dim)

        PT_DECLARE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                row_stack)

        PT_DECLARE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                column_stack)

        PT_DECLARE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                hstack)

        PT_DECLARE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                vstack)

        // many to many
        PT_DECLARE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                atleast_1d)

        PT_DECLARE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                atleast_2d)

        PT_DECLARE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                atleast_3d)

        PT_DECLARE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                broadcast_tensors)

        PT_DECLARE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                meshgrid)

        PT_DECLARE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                meshgrid, c10::string_view indexing)
    }
}

#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OP
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OP_WITH
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OP_WITH2
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OP_WITH3
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OP_
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OP__WITH
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OP__WITH2
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OP__WITH3
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH3
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3

#undef PT_DECLARE_ONE2MANY_PASSTHROUGH_OP
#undef PT_DECLARE_ONE2MANY_PASSTHROUGH_OP_WITH
#undef PT_DECLARE_ONE2MANY_PASSTHROUGH_OP_WITH2
#undef PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2

#undef PT_DECLARE_MANY2ONE_PASSTHROUGH_OP
#undef PT_DECLARE_MANY2ONE_PASSTHROUGH_OP_WITH
#undef PT_DECLARE_MANY2ONE_PASSTHROUGH_OP_WITH2
#undef PT_DECLARE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2

#undef PT_DECLARE_MANY2MANY_PASSTHROUGH_OP
#undef PT_DECLARE_MANY2MANY_PASSTHROUGH_OP_WITH
#undef PT_DECLARE_MANY2MANY_PASSTHROUGH_OP_WITH2
#undef PT_DECLARE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2
