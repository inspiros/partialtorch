#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "reduction.h"
#include "utils/fill_identity.h"
#include "utils/irepeat.h"
#include "utils/result_with_indices.h"
#include "utils/mask_utils.h"
#include "utils/schema_utils.h"

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<typename self_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> pad_impl(
                    const self_T &self,
                    c10::SymIntArrayRef pad,
                    c10::string_view mode,
                    c10::optional<double> value,
                    c10::string_view mask_mode,
                    c10::optional<bool> mask_value) {
                if (!utils::has_tensor_mask(self))
                    return masked_pair(at::_ops::pad::call(utils::get_data(self), pad, mode, value));

                auto output_data = at::_ops::pad::call(utils::get_data(self), pad, mode, value);
                auto output_mask = mask_value.has_value() ?
                                   at::_ops::pad::call(utils::get_tensor_mask(self), pad,
                                                       mask_mode, mask_value.value())
                                                          : at::_ops::pad::call(utils::get_tensor_mask(self), pad,
                                                                                mask_mode, {});
                return masked_pair(output_data, output_mask);
            }
        }

        c10::intrusive_ptr<TensorMaskedPair> pad(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                c10::SymIntArrayRef pad,
                c10::string_view mode,
                c10::optional<double> value,
                c10::string_view mask_mode,
                c10::optional<bool> mask_value) {
            return impl::pad_impl(self, pad, mode, value, mask_mode, mask_value);
        }

        c10::intrusive_ptr<TensorMaskedPair> pad(
                const at::Tensor &self,
                c10::SymIntArrayRef pad,
                c10::string_view mode,
                c10::optional<double> value,
                c10::string_view mask_mode,
                c10::optional<bool> mask_value) {
            return impl::pad_impl(self, pad, mode, value, mask_mode, mask_value);
        }

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            m.def(utils::FunctionSchemaBuilder("pad").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<c10::SymIntArrayRef>("pad")
                          .arg<c10::string_view>("mode", "\"constant\"")
                          .arg<c10::optional<double>>("value", "None")
                          .vararg()
                          .arg<c10::string_view>("mask_mode", "\"constant\"")
                          .arg<c10::optional<bool>>("mask_value", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair>(*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          c10::SymIntArrayRef,
                          c10::string_view,
                          c10::optional<double>,
                          c10::string_view,
                          c10::optional<bool>)>(pad)));
            m.def(utils::FunctionSchemaBuilder("pad").overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<c10::SymIntArrayRef>("pad")
                          .arg<c10::string_view>("mode", "\"constant\"")
                          .arg<c10::optional<double>>("value", "None")
                          .vararg()
                          .arg<c10::string_view>("mask_mode", "\"constant\"")
                          .arg<c10::optional<bool>>("mask_value", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair>(*)(
                          const at::Tensor &,
                          c10::SymIntArrayRef,
                          c10::string_view,
                          c10::optional<double>,
                          c10::string_view,
                          c10::optional<bool>)>(pad)));
        }
    }
}
