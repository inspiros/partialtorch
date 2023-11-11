#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "masked_adaptive_avg_pool.h"
#include "masked_avg_pool.h"
#include "reduction.h"
#include "utils/fill_identity.h"
#include "utils/irepeat.h"
#include "utils/result_with_indices.h"
#include "utils/mask_utils.h"
#include "utils/schema_utils.h"

// ~~~~~ ops definition macros ~~~~~
#define PT_DEFINE_UPSAMPLE_NEAREST_OP_WITH(NAME, IMPL_OP, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                          \
        SELF_T self,                                                                                \
        OUTPUT_SIZE_T output_size,                                                                  \
        ARG1_T ARG1_NAME) {                                                                         \
    static constexpr auto op = IMPL_OP;                                                             \
    return impl::upsample_nearest_impl(                                                             \
            op, self, output_size, ARG1_NAME);                                                      \
}

#define PT_DEFINE_UPSAMPLE_NEAREST_OP_WITH2(NAME, IMPL_OP, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                              \
        SELF_T self,                                                                                                    \
        OUTPUT_SIZE_T output_size,                                                                                      \
        ARG1_T ARG1_NAME,                                                                                               \
        ARG2_T ARG2_NAME) {                                                                                             \
    static constexpr auto op = IMPL_OP;                                                                                 \
    return impl::upsample_nearest_impl(                                                                                 \
            op, self, output_size, ARG1_NAME, ARG2_NAME);                                                               \
}

#define PT_DEFINE_UPSAMPLE_NEAREST_OP_WITH3(NAME, IMPL_OP, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                                                 \
        SELF_T self,                                                                                                                       \
        OUTPUT_SIZE_T output_size,                                                                                                         \
        ARG1_T ARG1_NAME,                                                                                                                  \
        ARG2_T ARG2_NAME,                                                                                                                  \
        ARG3_T ARG3_NAME) {                                                                                                                \
    static constexpr auto op = IMPL_OP;                                                                                                    \
    return impl::upsample_nearest_impl(                                                                                                    \
            op, self, output_size, ARG1_NAME, ARG2_NAME, ARG3_NAME);                                                                       \
}

#define PT_DEFINE_UPSAMPLE_INTERP_OP_WITH(NAME, IMPL_OP, MASK_OP, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                  \
        SELF_T self,                                                                                        \
        OUTPUT_SIZE_T output_size,                                                                          \
        ARG1_T ARG1_NAME) {                                                                                 \
    static constexpr auto op = IMPL_OP;                                                                     \
    static constexpr auto mask_op = MASK_OP;                                                                \
    return impl::upsample_interp_impl(                                                                      \
            op, mask_op, self, output_size, ARG1_NAME);                                                     \
}

#define PT_DEFINE_UPSAMPLE_INTERP_OP_WITH2(NAME, IMPL_OP, MASK_OP, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                                      \
        SELF_T self,                                                                                                            \
        OUTPUT_SIZE_T output_size,                                                                                              \
        ARG1_T ARG1_NAME,                                                                                                       \
        ARG2_T ARG2_NAME) {                                                                                                     \
    static constexpr auto op = IMPL_OP;                                                                                         \
    static constexpr auto mask_op = MASK_OP;                                                                                    \
    return impl::upsample_interp_impl(                                                                                          \
            op, mask_op, self, output_size, ARG1_NAME, ARG2_NAME);                                                              \
}

#define PT_DEFINE_UPSAMPLE_INTERP_OP_WITH3(NAME, IMPL_OP, MASK_OP, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                                                         \
        SELF_T self,                                                                                                                               \
        OUTPUT_SIZE_T output_size,                                                                                                                 \
        ARG1_T ARG1_NAME,                                                                                                                          \
        ARG2_T ARG2_NAME,                                                                                                                          \
        ARG3_T ARG3_NAME) {                                                                                                                        \
    static constexpr auto op = IMPL_OP;                                                                                                            \
    static constexpr auto mask_op = MASK_OP;                                                                                                       \
    return impl::upsample_interp_impl(                                                                                                             \
            op, mask_op, self, output_size, ARG1_NAME, ARG2_NAME, ARG3_NAME);                                                                      \
}

#define PT_DEFINE_UPSAMPLE_INTERP_OP_WITH4(NAME, IMPL_OP, MASK_OP, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                                                                            \
        SELF_T self,                                                                                                                                                  \
        OUTPUT_SIZE_T output_size,                                                                                                                                    \
        ARG1_T ARG1_NAME,                                                                                                                                             \
        ARG2_T ARG2_NAME,                                                                                                                                             \
        ARG3_T ARG3_NAME,                                                                                                                                             \
        ARG4_T ARG4_NAME) {                                                                                                                                           \
    static constexpr auto op = IMPL_OP;                                                                                                                               \
    static constexpr auto mask_op = MASK_OP;                                                                                                                          \
    return impl::upsample_interp_impl(                                                                                                                                \
            op, mask_op, self, output_size, ARG1_NAME, ARG2_NAME, ARG3_NAME, ARG4_NAME);                                                                              \
}

#define PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OP_WITH(NAME, IMPL_OP, MASK_OP, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                          \
        SELF_T self,                                                                                                \
        OUTPUT_SIZE_T output_size,                                                                                  \
        ARG1_T ARG1_NAME) {                                                                                         \
    static constexpr auto op = IMPL_OP;                                                                             \
    static constexpr auto mask_op = MASK_OP;                                                                        \
    return impl::partial_upsample_interp_impl(                                                                      \
            op, mask_op, self, output_size, ARG1_NAME);                                                             \
}

#define PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OP_WITH2(NAME, IMPL_OP, MASK_OP, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                                              \
        SELF_T self,                                                                                                                    \
        OUTPUT_SIZE_T output_size,                                                                                                      \
        ARG1_T ARG1_NAME,                                                                                                               \
        ARG2_T ARG2_NAME) {                                                                                                             \
    static constexpr auto op = IMPL_OP;                                                                                                 \
    static constexpr auto mask_op = MASK_OP;                                                                                            \
    return impl::partial_upsample_interp_impl(                                                                                          \
            op, mask_op, self, output_size, ARG1_NAME, ARG2_NAME);                                                                      \
}

#define PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OP_WITH3(NAME, IMPL_OP, MASK_OP, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                                                                 \
        SELF_T self,                                                                                                                                       \
        OUTPUT_SIZE_T output_size,                                                                                                                         \
        ARG1_T ARG1_NAME,                                                                                                                                  \
        ARG2_T ARG2_NAME,                                                                                                                                  \
        ARG3_T ARG3_NAME) {                                                                                                                                \
    static constexpr auto op = IMPL_OP;                                                                                                                    \
    static constexpr auto mask_op = MASK_OP;                                                                                                               \
    return impl::partial_upsample_interp_impl(                                                                                                             \
            op, mask_op, self, output_size, ARG1_NAME, ARG2_NAME, ARG3_NAME);                                                                              \
}

#define PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OP_WITH4(NAME, IMPL_OP, MASK_OP, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                                                                                    \
        SELF_T self,                                                                                                                                                          \
        OUTPUT_SIZE_T output_size,                                                                                                                                            \
        ARG1_T ARG1_NAME,                                                                                                                                                     \
        ARG2_T ARG2_NAME,                                                                                                                                                     \
        ARG3_T ARG3_NAME,                                                                                                                                                     \
        ARG4_T ARG4_NAME) {                                                                                                                                                   \
    static constexpr auto op = IMPL_OP;                                                                                                                                       \
    static constexpr auto mask_op = MASK_OP;                                                                                                                                  \
    return impl::partial_upsample_interp_impl(                                                                                                                                \
            op, mask_op, self, output_size, ARG1_NAME, ARG2_NAME, ARG3_NAME, ARG4_NAME);                                                                                      \
}

#define PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME) \
PT_DEFINE_UPSAMPLE_NEAREST_OP_WITH(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME) \
PT_DEFINE_UPSAMPLE_NEAREST_OP_WITH(NAME, IMPL_OP, const at::Tensor &, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME)

#define PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_UPSAMPLE_NEAREST_OP_WITH2(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_UPSAMPLE_NEAREST_OP_WITH2(NAME, IMPL_OP, const at::Tensor &, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, IMPL_OP, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_UPSAMPLE_NEAREST_OP_WITH3(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_UPSAMPLE_NEAREST_OP_WITH3(NAME, IMPL_OP, const at::Tensor &, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)

#define PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, MASK_OP, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_UPSAMPLE_INTERP_OP_WITH2(NAME, IMPL_OP, MASK_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_UPSAMPLE_INTERP_OP_WITH2(NAME, IMPL_OP, MASK_OP, const at::Tensor &, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, IMPL_OP, MASK_OP, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_UPSAMPLE_INTERP_OP_WITH3(NAME, IMPL_OP, MASK_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_UPSAMPLE_INTERP_OP_WITH3(NAME, IMPL_OP, MASK_OP, const at::Tensor &, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)

#define PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH4(NAME, IMPL_OP, MASK_OP, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME) \
PT_DEFINE_UPSAMPLE_INTERP_OP_WITH4(NAME, IMPL_OP, MASK_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME) \
PT_DEFINE_UPSAMPLE_INTERP_OP_WITH4(NAME, IMPL_OP, MASK_OP, const at::Tensor &, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME)

#define PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, MASK_OP, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OP_WITH2(NAME, IMPL_OP, MASK_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OP_WITH2(NAME, IMPL_OP, MASK_OP, const at::Tensor &, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, IMPL_OP, MASK_OP, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OP_WITH3(NAME, IMPL_OP, MASK_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OP_WITH3(NAME, IMPL_OP, MASK_OP, const at::Tensor &, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)

#define PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH4(NAME, IMPL_OP, MASK_OP, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME) \
PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OP_WITH4(NAME, IMPL_OP, MASK_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME) \
PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OP_WITH4(NAME, IMPL_OP, MASK_OP, const at::Tensor &, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME)

// ~~~~~ ops registration macros ~~~~~
#define PT_UPSAMPLE_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, SELF_T, OUTPUT_SIZE_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<SELF_T>("self").arg<OUTPUT_SIZE_T, DIMENSION>("output_size").ret<TensorMaskedPair>()

#define PT_REGISTER_UPSAMPLE_OP_WITH(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_UPSAMPLE_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, SELF_T, OUTPUT_SIZE_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OUTPUT_SIZE_T, ARG1_T)>(NAME)));

#define PT_REGISTER_UPSAMPLE_OP_WITH2(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_UPSAMPLE_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, SELF_T, OUTPUT_SIZE_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG2_T)>(NAME)));

#define PT_REGISTER_UPSAMPLE_OP_WITH3(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
m.def(PT_UPSAMPLE_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, SELF_T, OUTPUT_SIZE_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).arg<ARG3_T>(#ARG3_NAME, #ARG3_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG2_T, ARG3_T)>(NAME)));

#define PT_REGISTER_UPSAMPLE_OP_WITH4(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT) \
m.def(PT_UPSAMPLE_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, SELF_T, OUTPUT_SIZE_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).arg<ARG3_T>(#ARG3_NAME, #ARG3_DEFAULT).arg<ARG4_T>(#ARG4_NAME, #ARG4_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OUTPUT_SIZE_T, ARG1_T, ARG2_T, ARG3_T, ARG4_T)>(NAME)));

#define PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, DIMENSION, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_UPSAMPLE_OP_WITH(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_UPSAMPLE_OP_WITH(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, DIMENSION, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_UPSAMPLE_OP_WITH2(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_UPSAMPLE_OP_WITH2(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, POSTFIX, DIMENSION, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_UPSAMPLE_OP_WITH3(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_UPSAMPLE_OP_WITH3(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT)

#define PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH4(NAME, POSTFIX, DIMENSION, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT) \
PT_REGISTER_UPSAMPLE_OP_WITH4(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT) \
PT_REGISTER_UPSAMPLE_OP_WITH4(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &, OUTPUT_SIZE_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT)

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<typename op_T, typename self_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> upsample_nearest_impl(
                    op_T &&op,
                    const self_T &self,
                    Args &&... args) {
                if (!utils::has_tensor_mask(self))
                    return masked_pair(op.call(utils::get_data(self), args...));

                auto output_data = op.call(utils::get_data(self), args...);
                auto output_mask = op.call(utils::get_tensor_mask(self), args...);
                return masked_pair(output_data, output_mask);
            }

            template<typename op_T, typename mask_op_T, typename self_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> upsample_interp_impl(
                    op_T &&op,
                    mask_op_T &&mask_op,
                    const self_T &self,
                    Args &&... args) {
                if (!utils::has_tensor_mask(self))
                    return masked_pair(op.call(utils::get_data(self), args...));

                auto output_data = op.call(utils::get_data(self), args...);
                at::Tensor output_mask;
                if constexpr (std::is_same_v<std::base_t<mask_op_T>, nullptr_t>) {
                    {
                        at::NoGradGuard g;
                        output_mask = op.call(
                                utils::get_tensor_mask(self, output_data.options()), args...);
                    }
                    output_mask = output_mask.eq(1);
                } else {
                    output_mask = mask_op.call(
                            utils::get_tensor_mask(self), args...);
                }
                return masked_pair(output_data, output_mask);
            }

            template<typename op_T, typename mask_op_T, typename self_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_upsample_interp_impl(
                    op_T &&op,
                    mask_op_T &&mask_op,
                    const self_T &self,
                    Args &&... args) {
                if (!utils::has_tensor_mask(self))
                    return masked_pair(op.call(utils::get_data(self), args...));

                auto output_data = op.call(utils::_ops::fill_identity_zeros<false>::call(self), args...);
                at::Tensor output_mask;
                if constexpr (std::is_same_v<std::base_t<mask_op_T>, nullptr_t>) {
                    {
                        at::NoGradGuard g;
                        output_mask = op.call(
                                utils::get_tensor_mask(self, output_data.options()), args...);
                    }
                    output_mask = output_mask.ne(0);
                } else {
                    output_mask = mask_op.call(
                            utils::get_tensor_mask(self), args...);
                }
                return masked_pair(output_data, output_mask);
            }
        }

        // upsample_nearest
        PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                upsample_nearest1d, at::_ops::upsample_nearest1d_vec(), at::OptionalSymIntArrayRef,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                upsample_nearest1d, at::_ops::upsample_nearest1d(), c10::SymIntArrayRef,
                c10::optional<double>, scales)

        PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                _upsample_nearest_exact1d, at::_ops::_upsample_nearest_exact1d_vec(), at::OptionalSymIntArrayRef,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                _upsample_nearest_exact1d, at::_ops::_upsample_nearest_exact1d(), c10::SymIntArrayRef,
                c10::optional<double>, scales)

        PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                upsample_nearest2d, at::_ops::upsample_nearest2d_vec(), at::OptionalSymIntArrayRef,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                upsample_nearest2d, at::_ops::upsample_nearest2d(), c10::SymIntArrayRef,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                _upsample_nearest_exact2d, at::_ops::_upsample_nearest_exact2d_vec(), at::OptionalSymIntArrayRef,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                _upsample_nearest_exact2d, at::_ops::_upsample_nearest_exact2d(), c10::SymIntArrayRef,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                upsample_nearest3d, at::_ops::upsample_nearest3d_vec(), at::OptionalSymIntArrayRef,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                upsample_nearest3d, at::_ops::upsample_nearest3d(), c10::SymIntArrayRef,
                c10::optional<double>, scales_d,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                _upsample_nearest_exact3d, at::_ops::_upsample_nearest_exact3d_vec(), at::OptionalSymIntArrayRef,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_UPSAMPLE_NEAREST_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                _upsample_nearest_exact3d, at::_ops::_upsample_nearest_exact3d(), c10::SymIntArrayRef,
                c10::optional<double>, scales_d,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        // upsample_lerp
        PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                upsample_linear1d, at::_ops::upsample_linear1d_vec(), nullptr, at::OptionalSymIntArrayRef,
                bool, align_corners,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                upsample_linear1d, at::_ops::upsample_linear1d(), nullptr, c10::SymIntArrayRef,
                bool, align_corners,
                c10::optional<double>, scales)

        PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                upsample_bilinear2d, at::_ops::upsample_bilinear2d_vec(), nullptr, at::OptionalSymIntArrayRef,
                bool, align_corners,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                upsample_bilinear2d, at::_ops::upsample_bilinear2d(), nullptr, c10::SymIntArrayRef,
                bool, align_corners,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                _upsample_bilinear2d_aa, at::_ops::_upsample_bilinear2d_aa_vec(), nullptr, at::OptionalSymIntArrayRef,
                bool, align_corners,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                _upsample_bilinear2d_aa, at::_ops::_upsample_bilinear2d_aa(), nullptr, c10::SymIntArrayRef,
                bool, align_corners,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                upsample_trilinear3d, at::_ops::upsample_trilinear3d_vec(), nullptr, at::OptionalSymIntArrayRef,
                bool, align_corners,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH4(
                upsample_trilinear3d, at::_ops::upsample_trilinear3d(), nullptr, c10::SymIntArrayRef,
                bool, align_corners,
                c10::optional<double>, scales_d,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        // partial_upsample_lerp
        PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                partial_upsample_linear1d, at::_ops::upsample_linear1d_vec(), nullptr, at::OptionalSymIntArrayRef,
                bool, align_corners,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                partial_upsample_linear1d, at::_ops::upsample_linear1d(), nullptr, c10::SymIntArrayRef,
                bool, align_corners,
                c10::optional<double>, scales)

        PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                partial_upsample_bilinear2d, at::_ops::upsample_bilinear2d_vec(), nullptr, at::OptionalSymIntArrayRef,
                bool, align_corners,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                partial_upsample_bilinear2d, at::_ops::upsample_bilinear2d(), nullptr, c10::SymIntArrayRef,
                bool, align_corners,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                _partial_upsample_bilinear2d_aa, at::_ops::_upsample_bilinear2d_aa_vec(), nullptr, at::OptionalSymIntArrayRef,
                bool, align_corners,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                _partial_upsample_bilinear2d_aa, at::_ops::_upsample_bilinear2d_aa(), nullptr, c10::SymIntArrayRef,
                bool, align_corners,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                partial_upsample_trilinear3d, at::_ops::upsample_trilinear3d_vec(), nullptr, at::OptionalSymIntArrayRef,
                bool, align_corners,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH4(
                partial_upsample_trilinear3d, at::_ops::upsample_trilinear3d(), nullptr, c10::SymIntArrayRef,
                bool, align_corners,
                c10::optional<double>, scales_d,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        // upsample_bicubic
        PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                upsample_bicubic2d, at::_ops::upsample_bicubic2d_vec(), nullptr, at::OptionalSymIntArrayRef,
                bool, align_corners,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                upsample_bicubic2d, at::_ops::upsample_bicubic2d(), nullptr, c10::SymIntArrayRef,
                bool, align_corners,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                _upsample_bicubic2d_aa, at::_ops::_upsample_bicubic2d_aa_vec(), nullptr, at::OptionalSymIntArrayRef,
                bool, align_corners,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                _upsample_bicubic2d_aa, at::_ops::_upsample_bicubic2d_aa(), nullptr, c10::SymIntArrayRef,
                bool, align_corners,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        // partial_upsample_bicubic
        PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                partial_upsample_bicubic2d, at::_ops::upsample_bicubic2d_vec(), nullptr, at::OptionalSymIntArrayRef,
                bool, align_corners,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                partial_upsample_bicubic2d, at::_ops::upsample_bicubic2d(), nullptr, c10::SymIntArrayRef,
                bool, align_corners,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                _partial_upsample_bicubic2d_aa, at::_ops::_upsample_bicubic2d_aa_vec(), nullptr, at::OptionalSymIntArrayRef,
                bool, align_corners,
                c10::optional<at::ArrayRef<double>>, scale_factors)

        PT_DEFINE_PARTIAL_UPSAMPLE_INTERP_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                _partial_upsample_bicubic2d_aa, at::_ops::_upsample_bicubic2d_aa(), nullptr, c10::SymIntArrayRef,
                bool, align_corners,
                c10::optional<double>, scales_h,
                c10::optional<double>, scales_w)

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            // upsample_nearest
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    upsample_nearest1d, vec, 1, at::OptionalSymIntArrayRef,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    upsample_nearest1d, , 1, c10::SymIntArrayRef,
                    c10::optional<double>, scales, None)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    _upsample_nearest_exact1d, vec, 1, at::OptionalSymIntArrayRef,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    _upsample_nearest_exact1d, , 1, c10::SymIntArrayRef,
                    c10::optional<double>, scales, None)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    upsample_nearest2d, vec, 2, at::OptionalSymIntArrayRef,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    upsample_nearest2d, , 2, c10::SymIntArrayRef,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    _upsample_nearest_exact2d, vec, 2, at::OptionalSymIntArrayRef,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    _upsample_nearest_exact2d, , 2, c10::SymIntArrayRef,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    upsample_nearest3d, vec, 3, at::OptionalSymIntArrayRef,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    upsample_nearest3d, , 3, c10::SymIntArrayRef,
                    c10::optional<double>, scales_d, None,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    _upsample_nearest_exact3d, vec, 3, at::OptionalSymIntArrayRef,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    _upsample_nearest_exact3d, , 3, c10::SymIntArrayRef,
                    c10::optional<double>, scales_d, None,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)

            // upsample_lerp
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    upsample_linear1d, vec, 1, at::OptionalSymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    upsample_linear1d, , 1, c10::SymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<double>, scales, None)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    upsample_bilinear2d, vec, 2, at::OptionalSymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    upsample_bilinear2d, , 2, c10::SymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    _upsample_bilinear2d_aa, vec, 2, at::OptionalSymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    _upsample_bilinear2d_aa, , 2, c10::SymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    upsample_trilinear3d, vec, 3, at::OptionalSymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH4(
                    upsample_trilinear3d, , 3, c10::SymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<double>, scales_d, None,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)

            // partial_upsample_lerp
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    partial_upsample_linear1d, vec, 1, at::OptionalSymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    partial_upsample_linear1d, , 1, c10::SymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<double>, scales, None)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    partial_upsample_bilinear2d, vec, 2, at::OptionalSymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    partial_upsample_bilinear2d, , 2, c10::SymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    _partial_upsample_bilinear2d_aa, vec, 2, at::OptionalSymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    _partial_upsample_bilinear2d_aa, , 2, c10::SymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    partial_upsample_trilinear3d, vec, 3, at::OptionalSymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH4(
                    partial_upsample_trilinear3d, , 3, c10::SymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<double>, scales_d, None,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)

            // upsample_bicubic
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    upsample_bicubic2d, vec, 2, at::OptionalSymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    upsample_bicubic2d, , 2, c10::SymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    _upsample_bicubic2d_aa, vec, 2, at::OptionalSymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    _upsample_bicubic2d_aa, , 2, c10::SymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)

            // partial_upsample_bicubic
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    partial_upsample_bicubic2d, vec, 2, at::OptionalSymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    partial_upsample_bicubic2d, , 2, c10::SymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    _partial_upsample_bicubic2d_aa, vec, 2, at::OptionalSymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<at::ArrayRef<double>>, scale_factors,)
            PT_REGISTER_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    _partial_upsample_bicubic2d_aa, , 2, c10::SymIntArrayRef,
                    bool, align_corners, ,
                    c10::optional<double>, scales_h, None,
                    c10::optional<double>, scales_w, None)
        }
    }
}
