#include <torch/library.h>

#include "../MaskedPair.h"
#include "utils/fill_identity.h"
#include "utils/mask_utils.h"
#include "utils/schema_utils.h"
#include "utils/torch_library_utils.h"

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<typename input_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_einsum_impl(
                    c10::string_view equation,
                    const at::ArrayRef<input_T> &operands,
                    Args &&... args) {
                bool has_any_input_mask = false;
                for (const auto &input: operands)
                    has_any_input_mask |= utils::has_tensor_mask(input);
                if (!has_any_input_mask)
                    return masked_pair(at::einsum(equation, utils::get_data(operands), args...));

                static auto constexpr fill_identity_op = utils::_ops::fill_identity_zeros();
                std::vector<at::Tensor> tensors, masks;
                tensors.reserve(operands.size());
                for (const auto &input: operands) {
                    tensors.emplace_back(fill_identity_op.call(input));
                }

                auto output_data = at::einsum(equation, tensors, args...);
                auto mask_options = output_data.scalar_type();
                masks.reserve(operands.size());
                for (const auto &input: operands) {
                    masks.emplace_back(utils::get_tensor_mask(input, mask_options));
                }
                at::Tensor output_mask;
                {
                    at::NoGradGuard g;
                    output_mask = at::einsum(equation, masks, args...);
                }
                output_mask = output_mask.to(at::kBool);
                return masked_pair(output_data, output_mask);
            }
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_einsum(
                c10::string_view equation, TensorMaskedPairIntrusivePtrList tensors, at::OptionalIntArrayRef path) {
            return impl::partial_einsum_impl(equation, tensors, path);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_einsum(
                c10::string_view equation, at::TensorList tensors, at::OptionalIntArrayRef path) {
            return impl::partial_einsum_impl(equation, tensors, path);
        }

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            m.def(utils::FunctionSchemaBuilder("partial_einsum").overload("MaskedPairList")
                          .arg<c10::string_view>("equation")
                          .arg<TensorMaskedPairIntrusivePtrList>("tensors")
                          .vararg().arg<at::OptionalIntArrayRef>("path", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          c10::string_view,
                          TensorMaskedPairIntrusivePtrList,
                          at::OptionalIntArrayRef)>(partial_einsum)));
            m.def(utils::FunctionSchemaBuilder("partial_einsum").overload("TensorList")
                          .arg<c10::string_view>("equation")
                          .arg<at::TensorList>("tensors")
                          .vararg().arg<at::OptionalIntArrayRef>("path", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          c10::string_view,
                          at::TensorList,
                          at::OptionalIntArrayRef)>(partial_einsum)));
        }
    }
}
