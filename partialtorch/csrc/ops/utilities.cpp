#include <ATen/ATen.h>
#include <torch/types.h>

#include "../MaskedPair.h"
#include "utils/schema_utils.h"
#include "utils/reduction_utils.h"
#include "utils/torch_library_utils.h"

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<typename self_T, typename inputs_T>
            static C10_ALWAYS_INLINE void _backward_impl(
                    const self_T &self,
                    at::ArrayRef<inputs_T> inputs,
                    const c10::optional<at::Tensor> &gradient,
                    c10::optional<bool> retain_graph,
                    bool create_graph) {
                at::_ops::_backward::call(
                        utils::get_data(self),
                        utils::get_data(inputs),
                        gradient.has_value() && gradient->defined() && utils::has_tensor_mask(self) ? at::where(
                                utils::get_mask(self).value(), gradient.value(), 0) : gradient,
                        retain_graph, create_graph);
            }
        }

        void _backward(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                       TensorMaskedPairIntrusivePtrList inputs,
                       const c10::optional<at::Tensor> &gradient,
                       c10::optional<bool> retain_graph,
                       bool create_graph) {
            return impl::_backward_impl(self, inputs, gradient, retain_graph, create_graph);
        }

        void _backward(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                       at::TensorList inputs,
                       const c10::optional<at::Tensor> &gradient,
                       c10::optional<bool> retain_graph,
                       bool create_graph) {
            return impl::_backward_impl(self, inputs, gradient, retain_graph, create_graph);
        }

        void _backward(const at::Tensor &self,
                       TensorMaskedPairIntrusivePtrList inputs,
                       const c10::optional<at::Tensor> &gradient,
                       c10::optional<bool> retain_graph,
                       bool create_graph) {
            return impl::_backward_impl(self, inputs, gradient, retain_graph, create_graph);
        }

        void _backward(const at::Tensor &self,
                       at::TensorList inputs,
                       const c10::optional<at::Tensor> &gradient,
                       c10::optional<bool> retain_graph,
                       bool create_graph) {
            return impl::_backward_impl(self, inputs, gradient, retain_graph, create_graph);
        }

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            m.def(utils::FunctionSchemaBuilder("_backward").overload("MaskedPair_MaskedPairList")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<TensorMaskedPairIntrusivePtrList>("inputs")
                          .arg<const c10::optional<at::Tensor> &>("gradient", "None")
                          .arg<c10::optional<bool>>("retain_graph", "None")
                          .arg<bool>("create_graph", "False")
                          .ret<void>().schema().c_str(),
                  TORCH_FN(static_cast<void (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          TensorMaskedPairIntrusivePtrList,
                          const c10::optional<at::Tensor> &,
                          c10::optional<bool>,
                          bool)>(_backward)));
            m.def(utils::FunctionSchemaBuilder("_backward").overload("MaskedPair_TensorList")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<at::TensorList>("inputs")
                          .arg<const c10::optional<at::Tensor> &>("gradient", "None")
                          .arg<c10::optional<bool>>("retain_graph", "None")
                          .arg<bool>("create_graph", "False")
                          .ret<void>().schema().c_str(),
                  TORCH_FN(static_cast<void (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          at::TensorList,
                          const c10::optional<at::Tensor> &,
                          c10::optional<bool>,
                          bool)>(_backward)));
            m.def(utils::FunctionSchemaBuilder("_backward").overload("Tensor_MaskedPairList")
                          .arg<const at::Tensor &>("self")
                          .arg<TensorMaskedPairIntrusivePtrList>("inputs")
                          .arg<const c10::optional<at::Tensor> &>("gradient", "None")
                          .arg<c10::optional<bool>>("retain_graph", "None")
                          .arg<bool>("create_graph", "False")
                          .ret<void>().schema().c_str(),
                  TORCH_FN(static_cast<void (*)(
                          const at::Tensor &,
                          TensorMaskedPairIntrusivePtrList,
                          const c10::optional<at::Tensor> &,
                          c10::optional<bool>,
                          bool)>(_backward)));
            m.def(utils::FunctionSchemaBuilder("_backward").overload("Tensor_TensorList")
                          .arg<const at::Tensor &>("self")
                          .arg<at::TensorList>("inputs")
                          .arg<const c10::optional<at::Tensor> &>("gradient", "None")
                          .arg<c10::optional<bool>>("retain_graph", "None")
                          .arg<bool>("create_graph", "False")
                          .ret<void>().schema().c_str(),
                  TORCH_FN(static_cast<void (*)(
                          const at::Tensor &,
                          at::TensorList,
                          const c10::optional<at::Tensor> &,
                          c10::optional<bool>,
                          bool)>(_backward)));

            // multidim reduction
            m.def("partialtorch::_all(Tensor input, int[1]? dims=None, bool keepdim=False) -> Tensor",
                  TORCH_FN(static_cast<at::Tensor (*)(
                          const at::Tensor &, at::OptionalIntArrayRef, bool)>(utils::all)));
            m.def("partialtorch::_any(Tensor input, int[1]? dims=None, bool keepdim=False) -> Tensor",
                  TORCH_FN(static_cast<at::Tensor (*)(
                          const at::Tensor &, at::OptionalIntArrayRef, bool)>(utils::any)));
            m.def("partialtorch::_prod(Tensor input, int[1]? dims=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
                  TORCH_FN(static_cast<at::Tensor (*)(
                          const at::Tensor &, at::OptionalIntArrayRef, bool,
                          c10::optional<at::ScalarType>)>(utils::prod)));
        }
    }
}
