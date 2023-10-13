#pragma once

#include <ATen/ATen.h>

#include "../../MaskedPair.h"
#include "mask_utils.h"
#include "scalar_type_utils.h"
#include "type_utils.h"

namespace partialtorch {
    namespace ops {
        namespace utils {
            namespace _ops {
                namespace detail {
                    template<typename T, typename op_T, typename ... op_Ts>
                    C10_ALWAYS_INLINE auto compose_ops(T &&self) {
                        if constexpr (sizeof...(op_Ts) == 0)
                            return op_T::call(self);
                        else {
                            auto res = op_T::call(self);
                            return compose_ops<decltype(res), op_Ts...>(std::forward<decltype(res)>(res));
                        }
                    }

                    template<typename op_Ts, typename T, std::size_t... Is>
                    C10_ALWAYS_INLINE auto sequential_ops(const op_Ts &ops, T &&self, std::index_sequence<Is...>) {
                        if constexpr (sizeof...(Is) == 1)
                            return std::get<std::tuple_size_v<op_Ts> - 1>(ops).call(self);
                        else {
                            auto res = std::get<std::tuple_size_v<op_Ts> - sizeof...(Is)>(ops).call(self);
                            return sequential_ops(ops, std::forward<decltype(res)>(res),
                                                  std::make_index_sequence<sizeof...(Is) - 1>());
                        }
                    }
                }

/// \private
///
/// Compose multiple sequential ops.
                template<typename... Ts>
                struct compose {
                    template<typename T>
                    static auto call(T &&self) {
                        return detail::compose_ops<T, Ts...>(std::forward<T>(self));
                    }
                };

/// \private
///
/// Non-static variant of compose.
                template<typename... Ts>
                struct sequential {
                    std::tuple<Ts...> ops;

                    explicit sequential(const std::tuple<Ts...> &ops) : ops(ops) {}

                    explicit sequential(Ts &&... ops) : ops(std::make_tuple(ops...)) {}

                    template<typename T>
                    auto call(T &&self) const {
                        return detail::sequential_ops(ops, std::forward<T>(self),
                                                      std::make_index_sequence<sizeof...(Ts)>());
                    }
                };

/// \private
///
/// Cast tensor or TensorMaskedPair to dtype, with dtype being a template parameter.
                template<at::ScalarType dtype>
                struct cast {
                    static c10::intrusive_ptr<TensorMaskedPair> call(
                            const c10::intrusive_ptr<TensorMaskedPair> &self) {
                        return masked_pair(self->data_.to(dtype), self->mask_);
                    }

                    static at::Tensor call(const at::Tensor &self) {
                        return self.to(dtype);
                    }

                    template<typename T>
                    static std::vector<T> call(at::ArrayRef<T> inputs) {
                        std::vector<T> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input));
                        }
                        return outputs;
                    }
                };

/// \private
///
/// Cast tensor or TensorMaskedPair to dtype, with dtype being a template input
/// parameter to call method.
                template<bool copy = false>
                struct cast_dtype {
                    static c10::intrusive_ptr<TensorMaskedPair> call(
                            const c10::intrusive_ptr<TensorMaskedPair> &self,
                            const c10::optional<at::ScalarType> &dtype) {
                        if (dtype.has_value())
                            return masked_pair(self->data_.to(dtype.value()), self->mask_);
                        if constexpr (copy)
                            return self->clone();
                        else
                            return self;
                    }

                    static at::Tensor call(const at::Tensor &self,
                                           const c10::optional<at::ScalarType> &dtype) {
                        if (dtype.has_value())
                            return self.to(dtype.value());
                        if constexpr (copy)
                            return self.clone();
                        else
                            return self;
                    }

                    template<typename T>
                    static std::vector<T> call(at::ArrayRef<T> inputs,
                            const c10::optional<at::ScalarType> &dtype) {
                        std::vector<T> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input, dtype));
                        }
                        return outputs;
                    }
                };

/// \private
///
/// Cast tensor or TensorMaskedPair to dtype, with dtype being a state.
                template<bool copy = false>
                struct cast_state {
                    c10::optional<at::ScalarType> dtype;

                    cast_state(c10::optional<at::ScalarType> dtype = {}) : dtype(dtype) {}

                    C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> call(
                            const c10::intrusive_ptr<TensorMaskedPair> &self) const {
                        if (dtype.has_value())
                            return masked_pair(self->data_.to(dtype.value()), self->mask_);
                        if constexpr (copy)
                            return self->clone();
                        else
                            return self;
                    }

                    C10_ALWAYS_INLINE at::Tensor call(const at::Tensor &self) const {
                        if (dtype.has_value())
                            return self.to(dtype.value());
                        if constexpr (copy)
                            return self.clone();
                        else
                            return self;
                    }

                    template<typename T>
                    C10_ALWAYS_INLINE std::vector<T> call(at::ArrayRef<T> inputs) const {
                        std::vector<T> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input, dtype));
                        }
                        return outputs;
                    }
                };
            }  // namespace _ops

/// \private
///
/// Fill identity values at masked positions. The identity value is defined as
/// the initial value to reduction operation that has a property ``op(op_identity,
/// value) == value`` for any value in the domain of the operation.
            template<bool copy = false, typename T>
            C10_ALWAYS_INLINE at::Tensor fill_identity(
                    const T &self,
                    const at::Scalar &value) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<T>, TensorMaskedPair>) {
                    if (!get_mask(self).has_value())
                        return get_data<copy>(self);
                    return at::masked_fill(get_data(self), get_mask(self).value().logical_not(), value);
                } else if constexpr (std::is_same_v<T, at::Tensor>) {
                    return self;
                }
            }

/// \private
///
/// Inplace variant of fill_identity.
            template<typename T>
            C10_ALWAYS_INLINE at::Tensor fill_identity_(
                    T &self,
                    const at::Scalar &value) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<T>, TensorMaskedPair>) {
                    if (!get_mask(self).has_value())
                        return get_data(self);
                    return get_data(self).masked_fill_(get_mask(self).value().logical_not(), value);
                } else if constexpr (std::is_same_v<T, at::Tensor>) {
                    return self;
                }
            }

/// \private
///
/// Fill infinity or max values for integral types at masked positions.
            template<bool copy = false, typename T>
            C10_ALWAYS_INLINE at::Tensor fill_identity_posinf(
                    const T &self) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<T>, TensorMaskedPair>) {
                    if (!get_mask(self).has_value())
                        return get_data<copy>(self);
                    auto output_data = get_data<true>(self);
                    AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16,
                                               self->data_.scalar_type(), "fill_identity_posinf", ([&]() {
                        output_data.masked_fill_(get_mask(self).value().logical_not(),
                                                 CPPTypeLimits<scalar_t>::upper_bound());
                    }));
                    return output_data;
                } else if constexpr (std::is_same_v<T, at::Tensor>) {
                    return self;
                }
            }

/// \private
///
/// Inplace variant of fill_identity_posinf.
            template<typename T>
            C10_ALWAYS_INLINE at::Tensor fill_identity_posinf_(
                    T &self) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<T>, TensorMaskedPair>) {
                    if (!get_mask(self).has_value())
                        return get_data(self);
                    auto output_data = get_data(self);
                    AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16,
                                               output_data.scalar_type(), "fill_identity_posinf_", ([&]() {
                        output_data.masked_fill_(get_mask(self).value().logical_not(),
                                                 CPPTypeLimits<scalar_t>::upper_bound());
                    }));
                    return output_data;
                } else if constexpr (std::is_same_v<T, at::Tensor>) {
                    return self;
                }
            }

/// \private
///
/// Fill negative infinity or min values for integral types at masked positions.
            template<bool copy = false, typename T>
            C10_ALWAYS_INLINE at::Tensor fill_identity_neginf(
                    const T &self) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<T>, TensorMaskedPair>) {
                    if (!get_mask(self).has_value())
                        return get_data<copy>(self);
                    auto output_data = get_data<true>(self);
                    AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16,
                                               self->data_.scalar_type(), "fill_identity_neginf", ([&]() {
                        output_data.masked_fill_(get_mask(self).value().logical_not(),
                                                 CPPTypeLimits<scalar_t>::lower_bound());
                    }));
                    return output_data;
                } else if constexpr (std::is_same_v<T, at::Tensor>) {
                    return self;
                }
            }

/// \private
///
/// Inplace variant of fill_identity_neginf.
            template<typename T>
            C10_ALWAYS_INLINE at::Tensor fill_identity_neginf_(
                    T &self) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<T>, TensorMaskedPair>) {
                    if (!get_mask(self).has_value())
                        return get_data(self);
                    auto output_data = get_data(self);
                    AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16,
                                               output_data.scalar_type(), "fill_identity_neginf_", ([&]() {
                        output_data.masked_fill_(get_mask(self).value().logical_not(),
                                                 CPPTypeLimits<scalar_t>::lower_bound());
                    }));
                    return output_data;
                } else if constexpr (std::is_same_v<T, at::Tensor>) {
                    return self;
                }
            }

/// \private
///
/// Fill quiet nan at masked positions.
            template<bool copy = false, bool signaling = false, typename T>
            C10_ALWAYS_INLINE at::Tensor fill_identity_nan(
                    const T &self) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<T>, TensorMaskedPair>) {
                    if (!get_mask(self).has_value())
                        return get_data<copy>(self);
                    auto output_data = get_data<true>(self);
                    AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16,
                                               self->data_.scalar_type(), "fill_identity_nan", ([&]() {
                        constexpr scalar_t nan = signaling ? CPPTypeLimits<scalar_t>::signaling_nan()
                                                           : CPPTypeLimits<scalar_t>::quiet_nan();
                        output_data.masked_fill_(get_mask(self).value().logical_not(), nan);
                    }));
                    return output_data;
                } else if constexpr (std::is_same_v<T, at::Tensor>) {
                    return self;
                }
            }

/// \private
///
/// Inplace variant of fill_identity_neginf.
            template<bool signaling = false, typename T>
            C10_ALWAYS_INLINE at::Tensor fill_identity_nan_(
                    T &self) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<T>, TensorMaskedPair>) {
                    if (!get_mask(self).has_value())
                        return get_data(self);
                    auto output_data = get_data(self);
                    AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16,
                                               output_data.scalar_type(), "fill_identity_nan_", ([&]() {
                        constexpr scalar_t nan = signaling ? CPPTypeLimits<scalar_t>::signaling_nan()
                                                           : CPPTypeLimits<scalar_t>::quiet_nan();
                        output_data.masked_fill_(get_mask(self).value().logical_not(), nan);
                    }));
                    return output_data;
                } else if constexpr (std::is_same_v<T, at::Tensor>) {
                    return self;
                }
            }

            namespace _ops {
                template<bool copy = false>
                struct [[maybe_unused]] fill_identity_value {
                    static at::Tensor call(const c10::intrusive_ptr<TensorMaskedPair> &self,
                                           const at::Scalar &value) {
                        return utils::fill_identity<copy>(self, value);
                    }

                    template<typename T>
                    static T call(const T &self,
                                  const at::Scalar &) {
                        return utils::get_data<copy>(self);
                    }

                    template<typename T>
                    static std::vector<at::Tensor> call(at::ArrayRef<T> inputs) {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input));
                        }
                        return outputs;
                    }
                };

                struct [[maybe_unused]] fill_identity_value_ {
                    static at::Tensor call(c10::intrusive_ptr<TensorMaskedPair> &self,
                                           const at::Scalar &value) {
                        return utils::fill_identity_(self, value);
                    }

                    template<typename T>
                    static T call(const T &self,
                                  const at::Scalar &) {
                        return self;
                    }

                    template<typename T>
                    static std::vector<at::Tensor> call(at::ArrayRef<T> inputs) {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input));
                        }
                        return outputs;
                    }
                };

                template<bool copy = false>
                struct [[maybe_unused]] fill_identity_state_value {
                    at::Scalar value;

                    explicit fill_identity_state_value(const at::Scalar &value) : value(value) {}

                    C10_ALWAYS_INLINE at::Tensor call(const c10::intrusive_ptr<TensorMaskedPair> &self) const {
                        return utils::fill_identity<copy>(self, value);
                    }

                    template<typename T>
                    C10_ALWAYS_INLINE T call(const T &self) const {
                        return utils::get_data<copy>(self);
                    }

                    template<typename T>
                    C10_ALWAYS_INLINE std::vector<at::Tensor> call(at::ArrayRef<T> inputs) const {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input, value));
                        }
                        return outputs;
                    }
                };

                struct [[maybe_unused]] fill_identity_state_value_ {
                    at::Scalar value;

                    explicit fill_identity_state_value_(const at::Scalar &value) : value(value) {}

                    C10_ALWAYS_INLINE at::Tensor call(c10::intrusive_ptr<TensorMaskedPair> &self) const {
                        return utils::fill_identity_(self, value);
                    }

                    template<typename T>
                    C10_ALWAYS_INLINE T call(const T &self) const {
                        return self;
                    }

                    template<typename T>
                    C10_ALWAYS_INLINE std::vector<at::Tensor> call(at::ArrayRef<T> inputs) const {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input, value));
                        }
                        return outputs;
                    }
                };

                template<bool copy = false>
                struct [[maybe_unused]] fill_identity_zeros {
                    static at::Tensor call(const c10::intrusive_ptr<TensorMaskedPair> &self) {
                        return utils::fill_identity<copy>(self, 0);
                    }

                    template<typename T>
                    static T call(const T &self) {
                        return utils::get_data<copy>(self);
                    }

                    template<typename T>
                    static std::vector<at::Tensor> call(at::ArrayRef<T> inputs) {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input));
                        }
                        return outputs;
                    }
                };

                struct [[maybe_unused]] fill_identity_zeros_ {
                    static at::Tensor call(c10::intrusive_ptr<TensorMaskedPair> &self) {
                        return utils::fill_identity_(self, 0);
                    }

                    template<typename T>
                    static T call(const T &self) {
                        return self;
                    }

                    template<typename T>
                    static std::vector<at::Tensor> call(at::ArrayRef<T> inputs) {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input));
                        }
                        return outputs;
                    }
                };

                template<bool copy = false>
                struct [[maybe_unused]] fill_identity_ones {
                    static at::Tensor call(const c10::intrusive_ptr<TensorMaskedPair> &self) {
                        return utils::fill_identity<copy>(self, 1);
                    }

                    template<typename T>
                    static T call(const T &self) {
                        return utils::get_data<copy>(self);
                    }

                    template<typename T>
                    static std::vector<at::Tensor> call(at::ArrayRef<T> inputs) {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input));
                        }
                        return outputs;
                    }
                };

                struct [[maybe_unused]] fill_identity_ones_ {
                    static at::Tensor call(c10::intrusive_ptr<TensorMaskedPair> &self) {
                        return utils::fill_identity_(self, 1);
                    }

                    template<typename T>
                    static T call(const T &self) {
                        return self;
                    }

                    template<typename T>
                    static std::vector<at::Tensor> call(at::ArrayRef<T> inputs) {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input));
                        }
                        return outputs;
                    }
                };

                template<bool copy = false>
                struct [[maybe_unused]] fill_identity_posinf {
                    static at::Tensor call(const c10::intrusive_ptr<TensorMaskedPair> &self) {
                        return utils::fill_identity_posinf<copy>(self);
                    }

                    template<typename T>
                    static T call(const T &self) {
                        return utils::get_data<copy>(self);
                    }

                    template<typename T>
                    static std::vector<at::Tensor> call(at::ArrayRef<T> inputs) {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input));
                        }
                        return outputs;
                    }
                };

                struct [[maybe_unused]] fill_identity_posinf_ {
                    static at::Tensor call(c10::intrusive_ptr<TensorMaskedPair> &self) {
                        return utils::fill_identity_posinf_(self);
                    }

                    template<typename T>
                    static T call(const T &self) {
                        return self;
                    }

                    template<typename T>
                    static std::vector<at::Tensor> call(at::ArrayRef<T> inputs) {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input));
                        }
                        return outputs;
                    }
                };

                template<bool copy = false>
                struct [[maybe_unused]] fill_identity_neginf {
                    static at::Tensor call(const c10::intrusive_ptr<TensorMaskedPair> &self) {
                        return utils::fill_identity_neginf<copy>(self);
                    }

                    template<typename T>
                    static T call(const T &self) {
                        return utils::get_data<copy>(self);
                    }

                    template<typename T>
                    static std::vector<at::Tensor> call(at::ArrayRef<T> inputs) {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input));
                        }
                        return outputs;
                    }
                };

                struct [[maybe_unused]] fill_identity_neginf_ {
                    static at::Tensor call(c10::intrusive_ptr<TensorMaskedPair> &self) {
                        return utils::fill_identity_neginf_(self);
                    }

                    template<typename T>
                    static T call(const T &self) {
                        return self;
                    }

                    template<typename T>
                    static std::vector<at::Tensor> call(at::ArrayRef<T> inputs) {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input));
                        }
                        return outputs;
                    }
                };

                template<bool copy = false, bool signaling = false>
                struct [[maybe_unused]] fill_identity_nan {
                    static at::Tensor call(const c10::intrusive_ptr<TensorMaskedPair> &self) {
                        TORCH_CHECK(!at::isIntegralType(self->data_.scalar_type(), true),
                                    "Only floating point tensors can be filled with nan. Got ",
                                    self->data_.scalar_type())
                        return utils::fill_identity_nan<copy, signaling>(self);
                    }

                    template<typename T>
                    static T call(const T &self) {
                        return utils::get_data<copy>(self);
                    }

                    template<typename T>
                    static std::vector<at::Tensor> call(at::ArrayRef<T> inputs) {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input));
                        }
                        return outputs;
                    }
                };

                template<bool signaling = false>
                struct [[maybe_unused]] fill_identity_nan_ {
                    static at::Tensor call(c10::intrusive_ptr<TensorMaskedPair> &self) {
                        TORCH_CHECK(at::is_floating_point(self->data_),
                                    "Only floating point tensors can be filled with nan. Got ",
                                    self->data_.scalar_type())
                        return utils::fill_identity_nan_<signaling>(self);
                    }

                    template<typename T>
                    static T call(const T &self) {
                        return self;
                    }

                    template<typename T>
                    static std::vector<at::Tensor> call(at::ArrayRef<T> inputs) {
                        std::vector<at::Tensor> outputs;
                        outputs.reserve(inputs.size());
                        for (const auto &input: inputs) {
                            outputs.emplace_back(call(input));
                        }
                        return outputs;
                    }
                };
            }
        }
    }
}
