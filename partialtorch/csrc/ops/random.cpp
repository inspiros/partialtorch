#include <ATen/ATen.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "utils/schema_utils.h"
#include "utils/torch_library_utils.h"

namespace partialtorch {
    namespace ops {
        c10::intrusive_ptr<TensorMaskedPair> rand_mask(
                const at::Tensor &data,
                double p,
                const c10::optional<at::Scalar> &mask_value,
                c10::optional<at::Generator> generator) {
            auto options = at::TensorOptions().dtype<double>().device(data.device());
            auto mask = at::bernoulli(at::full(data.sizes(), p, options), generator).to(at::kBool);
            return masked_pair(mask_value.has_value() ? at::where(mask, data, mask_value.value())
                                                      : data.clone(), mask);
        }

        c10::intrusive_ptr<TensorMaskedPair> rand_mask(
                const at::Tensor &data,
                const at::Tensor &p,
                const c10::optional<at::Scalar> &mask_value,
                c10::optional<at::Generator> generator) {
            auto mask = at::bernoulli(p, generator).to(at::kBool);
            return masked_pair(mask_value.has_value() ? at::where(mask, data, mask_value.value())
                                                      : data.clone(), mask);
        }

        c10::intrusive_ptr<TensorMaskedPair> rand_mask(
                const_intrusive_ptr_arg_t<TensorMaskedPair> pair,
                double p,
                const c10::optional<at::Scalar> &mask_value,
                c10::optional<at::Generator> generator) {
            auto options = at::TensorOptions().dtype<double>().device(pair->data_.device());
            auto mask = at::bernoulli(at::full(pair->data_.sizes(), p, options), generator).to(at::kBool);
            return masked_pair(mask_value.has_value() ? at::where(mask, pair->data_, mask_value.value())
                                                      : pair->data_.clone(), mask);
        }

        c10::intrusive_ptr<TensorMaskedPair> rand_mask(
                const_intrusive_ptr_arg_t<TensorMaskedPair> pair,
                const at::Tensor &p,
                const c10::optional<at::Scalar> &mask_value,
                c10::optional<at::Generator> generator) {
            auto mask = at::bernoulli(p, generator).to(at::kBool);
            return masked_pair(mask_value.has_value() ? at::where(mask, pair->data_, mask_value.value())
                                                      : pair->data_.clone(), mask);
        }

        c10::intrusive_ptr<TensorMaskedPair> rand_mask_(
                const at::Tensor &data,
                double p,
                const c10::optional<at::Scalar> &mask_value,
                c10::optional<at::Generator> generator) {
            auto options = at::TensorOptions().dtype<double>().device(data.device());
            auto mask = at::bernoulli(at::full(data.sizes(), p, options), generator).to(at::kBool);
            if (mask_value.has_value())
                data.masked_fill_(mask.logical_not(), mask_value.value());
            return masked_pair(data, mask);
        }

        c10::intrusive_ptr<TensorMaskedPair> rand_mask_(
                const at::Tensor &data,
                const at::Tensor &p,
                const c10::optional<at::Scalar> &mask_value,
                c10::optional<at::Generator> generator) {
            auto mask = at::bernoulli(p, generator).to(at::kBool);
            if (mask_value.has_value())
                data.masked_fill_(mask.logical_not(), mask_value.value());
            return masked_pair(data, mask);
        }

        c10::intrusive_ptr<TensorMaskedPair> rand_mask_(
                intrusive_ptr_arg_t<TensorMaskedPair> pair,
                double p,
                const c10::optional<at::Scalar> &mask_value,
                c10::optional<at::Generator> generator) {
            auto options = at::TensorOptions().dtype<double>().device(pair->data_.device());
            pair->mask_ = at::bernoulli(at::full(pair->data_.sizes(), p, options), generator).to(at::kBool);
            if (mask_value.has_value())
                pair->fill_masked_(mask_value.value());
            return pair;
        }

        c10::intrusive_ptr<TensorMaskedPair> rand_mask_(
                intrusive_ptr_arg_t<TensorMaskedPair> pair,
                const at::Tensor &p,
                const c10::optional<at::Scalar> &mask_value,
                c10::optional<at::Generator> generator) {
            pair->mask_ = at::bernoulli(p, generator).to(at::kBool);
            if (mask_value.has_value())
                pair->fill_masked_(mask_value.value());
            return pair;
        }

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            m.def(utils::FunctionSchemaBuilder("rand_mask").add_overload("float")
                          .arg<const at::Tensor &>("input")
                          .arg<double>("p")
                          .arg<const c10::optional<at::Scalar> &>("mask_value", "None")
                          .vararg()
                          .arg<c10::optional<at::Generator>>("generator", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          double,
                          const c10::optional<at::Scalar> &,
                          c10::optional<at::Generator>)>(&rand_mask)));
            m.def(utils::FunctionSchemaBuilder("rand_mask").add_overload("Tensor")
                          .arg<const at::Tensor &>("input")
                          .arg<const at::Tensor &>("p")
                          .arg<const c10::optional<at::Scalar> &>("mask_value", "None")
                          .vararg()
                          .arg<c10::optional<at::Generator>>("generator", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Tensor &,
                          const c10::optional<at::Scalar> &,
                          c10::optional<at::Generator>)>(&rand_mask)));
            m.def(utils::FunctionSchemaBuilder("rand_mask").add_overload("MaskedPair_float")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input")
                          .arg<double>("p")
                          .arg<const c10::optional<at::Scalar> &>("mask_value", "None")
                          .vararg()
                          .arg<c10::optional<at::Generator>>("generator", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          double,
                          const c10::optional<at::Scalar> &,
                          c10::optional<at::Generator>)>(&rand_mask)));
            m.def(utils::FunctionSchemaBuilder("rand_mask").add_overload("MaskedPair_Tensor")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input")
                          .arg<const at::Tensor &>("p")
                          .arg<const c10::optional<at::Scalar> &>("mask_value", "None")
                          .vararg()
                          .arg<c10::optional<at::Generator>>("generator", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          const c10::optional<at::Scalar> &,
                          c10::optional<at::Generator>)>(&rand_mask)));

            m.def(utils::FunctionSchemaBuilder("rand_mask_").add_overload("float")
                          .arg<at::Tensor &>("input")
                          .arg<double>("p")
                          .arg<const c10::optional<at::Scalar> &>("mask_value", "None")
                          .vararg()
                          .arg<c10::optional<at::Generator>>("generator", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          double,
                          const c10::optional<at::Scalar> &,
                          c10::optional<at::Generator>)>(&rand_mask_)));
            m.def(utils::FunctionSchemaBuilder("rand_mask_").add_overload("Tensor")
                          .arg<at::Tensor &>("input")
                          .arg<const at::Tensor &>("p")
                          .arg<const c10::optional<at::Scalar> &>("mask_value", "None")
                          .vararg()
                          .arg<c10::optional<at::Generator>>("generator", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Tensor &,
                          const c10::optional<at::Scalar> &,
                          c10::optional<at::Generator>)>(&rand_mask_)));
            m.def(utils::FunctionSchemaBuilder("rand_mask_").add_overload("MaskedPair_float")
                          .arg<intrusive_ptr_arg_t<TensorMaskedPair>>("input")
                          .arg<double>("p")
                          .arg<const c10::optional<at::Scalar> &>("mask_value", "None")
                          .vararg()
                          .arg<c10::optional<at::Generator>>("generator", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          c10::intrusive_ptr<TensorMaskedPair>,
                          double,
                          const c10::optional<at::Scalar> &,
                          c10::optional<at::Generator>)>(&rand_mask_)));
            m.def(utils::FunctionSchemaBuilder("rand_mask_").add_overload("MaskedPair_Tensor")
                          .arg<intrusive_ptr_arg_t<TensorMaskedPair>>("input")
                          .arg<const at::Tensor &>("p")
                          .arg<const c10::optional<at::Scalar> &>("mask_value", "None")
                          .vararg()
                          .arg<c10::optional<at::Generator>>("generator", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          c10::intrusive_ptr<TensorMaskedPair>,
                          const at::Tensor &,
                          const c10::optional<at::Scalar> &,
                          c10::optional<at::Generator>)>(&rand_mask_)));
        }
    }
}
