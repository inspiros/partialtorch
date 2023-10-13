#pragma once

#include <ATen/ATen.h>

#include "../../MaskedPair.h"
#include "tensor_utils.h"
#include "type_utils.h"

namespace partialtorch {
    namespace ops {
        namespace utils {
            // These functions are used to standardize all inputs as pairs of data and mask.
            template<bool copy = false, typename T>
            C10_ALWAYS_INLINE T get_data(const c10::intrusive_ptr<MaskedPair<T>> &input) {
                if constexpr (copy && std::is_same_v<T, at::Tensor>) {
                    return input->data_.clone();
                } else {
                    return input->data_;
                }
            }

            template<bool copy = false, typename T>
            C10_ALWAYS_INLINE std::conditional_t<
                    std::is_same_v<T, at::Tensor>,
                    c10::optional<at::Tensor>,
                    c10::optional<bool>> get_mask(const c10::intrusive_ptr<MaskedPair<T>> &input) {
                if constexpr (copy && std::is_same_v<T, at::Tensor>) {
                    if (input->mask_.has_value())
                        return input->mask_->clone();
                    else
                        return input->mask_;
                } else {
                    return input->mask_;
                }
            }

            template<bool copy = false, typename T>
            C10_ALWAYS_INLINE T get_data(const MaskedPair<T> &input) {
                if constexpr (copy && std::is_same_v<T, at::Tensor>) {
                    return input.data_.clone();
                } else {
                    return input.data_;
                }
            }

            template<bool copy = false, typename T>
            C10_ALWAYS_INLINE std::conditional_t<
                    std::is_same_v<T, at::Tensor>,
                    c10::optional<at::Tensor>,
                    c10::optional<bool>> get_mask(const MaskedPair<T> &input) {
                if constexpr (copy && std::is_same_v<T, at::Tensor>) {
                    if (input.mask_.has_value())
                        return input.mask_->clone();
                    else
                        return input.mask_;
                } else {
                    return input.mask_;
                }
            }

            template<bool copy = false, typename T>
            C10_ALWAYS_INLINE T get_data(const T &input) {
                return input;
            }

            template<bool copy = false, typename T>
            C10_ALWAYS_INLINE std::conditional_t<
                    std::is_same_v<T, at::Tensor>,
                    c10::optional<at::Tensor>,
                    c10::optional<bool>> get_mask(const T &) {
                return {};
            }

            template<bool copy = false>
            C10_ALWAYS_INLINE std::vector<at::Tensor> get_data(
                    const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>> &inputs) {
                std::vector<at::Tensor> data;
                data.reserve(inputs.size());
                for (const auto &input: inputs) {
                    if constexpr (copy) {
                        data.emplace_back(input->data_.clone());
                    } else {
                        data.emplace_back(input->data_);
                    }
                }
                return data;
            }

            template<bool copy = false>
            C10_ALWAYS_INLINE std::vector<c10::optional<at::Tensor>> get_mask(
                    const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>> &inputs) {
                std::vector<c10::optional<at::Tensor>> masks;
                masks.reserve(inputs.size());
                for (const auto &input: inputs) {
                    if constexpr (copy) {
                        masks.emplace_back(input->mask_.has_value() ? input->mask_->clone() : input->mask_);
                    } else {
                        masks.emplace_back(input->mask_);
                    }
                }
                return masks;
            }

            template<bool copy = false>
            C10_ALWAYS_INLINE std::vector<at::Tensor> get_data(
                    const at::ArrayRef<at::Tensor> &inputs) {
                std::vector<at::Tensor> data;
                data.reserve(inputs.size());
                for (auto input: inputs) {
                    if constexpr (copy) {
                        data.emplace_back(input.clone());
                    } else {
                        data.emplace_back(input);
                    }
                }
                return data;
            }

            template<bool copy = false>
            C10_ALWAYS_INLINE std::vector<c10::optional<at::Tensor>> get_mask(
                    const at::ArrayRef<at::Tensor> &inputs) {
                std::vector<c10::optional<at::Tensor>> masks;
                masks.reserve(inputs.size());
                for (const auto _: c10::irange(inputs.size())) {
                    masks.emplace_back();
                }
                return masks;
            }

            template<bool copy = false>
            C10_ALWAYS_INLINE std::vector<at::Tensor> get_data(
                    const at::ITensorListRef &inputs) {
                std::vector<at::Tensor> data;
                data.reserve(inputs.size());
                for (auto input: inputs) {
                    if constexpr (copy) {
                        data.emplace_back(input.clone());
                    } else {
                        data.emplace_back(input);
                    }
                }
                return data;
            }

            template<bool copy = false>
            C10_ALWAYS_INLINE std::vector<c10::optional<at::Tensor>> get_mask(
                    const at::ITensorListRef &inputs) {
                std::vector<c10::optional<at::Tensor>> masks;
                masks.reserve(inputs.size());
                for (const auto _: c10::irange(inputs.size())) {
                    masks.emplace_back();
                }
                return masks;
            }

            // return mask as at::Tensor if possible
            template<bool copy = false>
            C10_ALWAYS_INLINE at::Tensor get_tensor_mask(
                    const c10::intrusive_ptr<TensorMaskedPair> &input,
                    c10::optional<at::TensorOptions> options = {}) {
                if (input->mask_.has_value()) {
                    if constexpr (copy)
                        return options.has_value() ? input->mask_->to(options.value())
                                                   : input->mask_->clone();
                    else
                        return options.has_value() ? input->mask_->to(options.value())
                                                   : input->mask_.value();
                } else
                    return at::ones_like(input->data_,
                                         options.value_or(at::TensorOptions().dtype(at::kBool)));
            }

            template<bool copy = false>
            C10_ALWAYS_INLINE at::Tensor get_tensor_mask(
                    const at::Tensor &input,
                    c10::optional<at::TensorOptions> options = {}) {
                return at::ones_like(input,
                                     options.value_or(at::TensorOptions().dtype(at::kBool)));
            }

            template<bool copy = false, typename T>
            C10_ALWAYS_INLINE std::vector<at::Tensor> get_tensor_mask(
                    at::ArrayRef<T> inputs,
                    c10::optional<at::TensorOptions> options = {}) {
                std::vector<at::Tensor> masks;
                masks.reserve(inputs.size());
                for (const auto &input: inputs) {
                    masks.emplace_back(get_tensor_mask(input));
                }
                return masks;
            }

            template<bool copy = false, typename T>
            C10_ALWAYS_INLINE at::Scalar get_tensor_mask(
                    const T &input,
                    c10::optional<at::TensorOptions> = {}) {
                return 1;
            }

            template<typename T>
            C10_ALWAYS_INLINE bool has_tensor_mask(const T &input) {
                auto mask = get_mask<false>(input);
                if constexpr (std::is_same_v<decltype(mask), c10::optional<at::Tensor>>) {
                    return mask.has_value();
                } else {
                    return false;
                }
            }

            template<typename T>
            C10_ALWAYS_INLINE bool has_tensor_mask(at::ArrayRef<T> inputs) {
                bool result = false;
                for (const auto &input: inputs)
                    result |= utils::has_tensor_mask(input);
                return result;
            }

            C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> &set_data(
                    c10::intrusive_ptr<TensorMaskedPair> &input,
                    const at::Tensor &data) {
                input->data_ = data;
                return input;
            }

            C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> &set_data_(
                    c10::intrusive_ptr<TensorMaskedPair> &input,
                    const at::Tensor &data) {
                input->data_.copy_(data);
                return input;
            }

            C10_ALWAYS_INLINE TensorMaskedPair &set_data(
                    TensorMaskedPair &input,
                    const at::Tensor &data) {
                input.data_ = data;
                return input;
            }

            C10_ALWAYS_INLINE TensorMaskedPair &set_data_(
                    TensorMaskedPair &input,
                    const at::Tensor &data) {
                input.data_.copy_(data);
                return input;
            }

            C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> &set_mask(
                    c10::intrusive_ptr<TensorMaskedPair> &input,
                    c10::optional<at::Tensor> mask) {
                input->mask_ = mask;
                return input;
            }

            C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> &set_mask_(
                    c10::intrusive_ptr<TensorMaskedPair> &input,
                    c10::optional<at::Tensor> mask) {
                if (input->mask_.has_value() && mask.has_value())
                    input->mask_->copy_(mask.value());
                else
                    input->mask_ = mask;
                return input;
            }

            C10_ALWAYS_INLINE TensorMaskedPair &set_mask(
                    TensorMaskedPair &input,
                    c10::optional<at::Tensor> mask) {
                input.mask_ = mask;
                return input;
            }

            C10_ALWAYS_INLINE TensorMaskedPair &set_mask_(
                    TensorMaskedPair &input,
                    c10::optional<at::Tensor> mask) {
                if (input.mask_.has_value() && mask.has_value())
                    input.mask_->copy_(mask.value());
                else
                    input.mask_ = mask;
                return input;
            }

/// \private
///
/// Return the non masked values of a Tensor.
            C10_ALWAYS_INLINE at::Tensor index_non_masked(
                    const at::Tensor &input,
                    c10::optional<at::Tensor> mask = {}) {
                if (!mask.has_value())
                    return input;
                return input.index({mask.value()});
            }

/// \private
///
/// Return the non masked values of a TensorMaskedPair. If input has no mask
/// or is a Tensor, return its flatten view.
            template<typename T>
            C10_ALWAYS_INLINE at::Tensor index_non_masked(const T &input) {
                if constexpr (std::is_same_v<T, c10::intrusive_ptr<TensorMaskedPair>>) {
                    if (input->mask_.has_value())
                        return input->data_.index({input->mask_.value()});
                    return input->data_.flatten();
                } else if constexpr (std::is_same_v<T, TensorMaskedPair>) {
                    if (input.mask_.has_value())
                        return input.data_.index({input.mask_.value()});
                    return input.data_.flatten();
                } else if constexpr (std::is_same_v<T, at::Tensor>) {
                    return input.flatten();
                } else {
                    static_assert(std::is_pod_v<T>, "T must be Tensor or MaskedPair<Tensor>.");
                }
            }

/// \private
///
/// Return the first tensor argument of binary ops. If none of the args is
/// of type tensor, return a undefined tensor if strict = false, otherwise
/// a runtime exception is raised.
            template<bool strict = true, typename self_T, typename other_T>
            C10_ALWAYS_INLINE at::Tensor get_any_tensor_data(
                    const self_T &self,
                    const other_T &other) {
                if constexpr (std::is_same_v<self_T, c10::intrusive_ptr<TensorMaskedPair>> ||
                              std::is_same_v<self_T, TensorMaskedPair> ||
                              std::is_same_v<self_T, at::Tensor>) {
                    return get_data(self);
                } else if constexpr (std::is_same_v<other_T, c10::intrusive_ptr<TensorMaskedPair>> ||
                                     std::is_same_v<other_T, TensorMaskedPair> ||
                                     std::is_same_v<other_T, at::Tensor>) {
                    return get_data(other);
                } else if constexpr (strict) {
                    TORCH_CHECK(false, "Either inputs must be of type tensor.")
                } else {
                    return {};
                }
            }

            namespace detail {
                struct get_any_tensor_wrapper {
                    at::Tensor arg;

                    template<typename T>
                    C10_ALWAYS_INLINE explicit get_any_tensor_wrapper(const T &arg) {
                        if constexpr (std::is_same_v<T, c10::intrusive_ptr<TensorMaskedPair>> ||
                                      std::is_same_v<T, TensorMaskedPair> ||
                                      std::is_same_v<T, at::Tensor>) {
                            this->arg = get_data(arg);
                        }
                    }

                    C10_ALWAYS_INLINE bool is_tensor() const {
                        return this->arg.defined();
                    }
                };

                C10_ALWAYS_INLINE get_any_tensor_wrapper operator||(
                        const get_any_tensor_wrapper &lhs, const get_any_tensor_wrapper &rhs) {
                    if (lhs.is_tensor())
                        return lhs;
                    else if (rhs.is_tensor())
                        return rhs;
                    else
                        return get_any_tensor_wrapper(get_any_tensor_data<false>(lhs, rhs));
                }
            }

/// \private
///
/// Return the first tensor argument of n-ary ops. If none of the args is
/// of type tensor, return a undefined tensor if strict = false, otherwise
/// a runtime exception is raised.
            template<bool strict = true, typename... Args>
            C10_ALWAYS_INLINE at::Tensor get_any_tensor_data(Args &&... args) {
                auto result = (... || detail::get_any_tensor_wrapper(args));
                if (result.is_tensor() || !strict) {
                    return result.arg;
                } else {
                    TORCH_CHECK(false, "Either inputs must be of type tensor.")
                }
            }

            // check
            namespace {
                C10_ALWAYS_INLINE std::vector<int64_t> _get_defined_inds(
                        at::ArrayRef<c10::optional<at::Tensor>> masks) {
                    std::vector<int64_t> defined_inds;
                    defined_inds.reserve(masks.size());
                    for (const auto i: at::irange(masks.size())) {
                        if (masks[i].has_value())
                            defined_inds.emplace_back(i);
                    }
                    return defined_inds;
                }
            }

            template<TensorMatchingCriteria criteria = BROADCASTABLE>
            C10_ALWAYS_INLINE bool masks_match(
                    const c10::optional<at::Tensor> &self_mask,
                    const c10::optional<at::Tensor> &other_mask) {
                if (!self_mask.has_value() || !other_mask.has_value())
                    return true;
                if constexpr (criteria == EXACT)
                    return at::equal(self_mask.value(), other_mask.value());
                else if constexpr (criteria == SHAPE)
                    return self_mask->sizes() == other_mask->sizes();
                else if constexpr (criteria == BROADCASTABLE)
                    return shapes_broadcastable(self_mask->sizes(), other_mask->sizes());
            }

            template<TensorMatchingCriteria criteria = BROADCASTABLE>
            C10_ALWAYS_INLINE bool masks_match(
                    at::ArrayRef<c10::optional<at::Tensor>> masks) {
                auto defined_inds = _get_defined_inds(masks);
                if (defined_inds.empty())
                    return true;
                for (auto iter = std::next(defined_inds.begin()); iter != defined_inds.end(); iter++) {
                    if (!masks_match<criteria>(masks[0], masks[*iter]))
                        return false;
                }
                return true;
            }

            namespace detail {
                template<TensorMatchingCriteria criteria = BROADCASTABLE>
                struct masks_match_wrapper {
                    const c10::optional<at::Tensor> mask;
                    const bool match;
                };

                template<TensorMatchingCriteria criteria>
                C10_ALWAYS_INLINE masks_match_wrapper<criteria> operator&(
                        const masks_match_wrapper<criteria> &lhs, const masks_match_wrapper<criteria> &rhs) {
                    if (!lhs.match || !rhs.match)
                        return {c10::optional<at::Tensor>{}, false};
                    else
                        return {rhs.mask, masks_match<criteria>(lhs.mask, rhs.mask)};
                }
            }

            template<TensorMatchingCriteria criteria = BROADCASTABLE, typename... mask_Ts>
            C10_ALWAYS_INLINE bool masks_match(mask_Ts &&... masks) {
                auto result = (... & detail::masks_match_wrapper<criteria>{masks, true});
                return result.match;
            }

            template<TensorMatchingCriteria criteria = BROADCASTABLE>
            C10_ALWAYS_INLINE void check_masks_match(
                    const c10::optional<at::Tensor> &self_mask,
                    const c10::optional<at::Tensor> &other_mask) {
                TORCH_CHECK(masks_match<criteria>(self_mask, other_mask),
                            "Masks do not match.")
            }

            template<TensorMatchingCriteria criteria = BROADCASTABLE>
            C10_ALWAYS_INLINE void check_masks_match(at::ArrayRef<c10::optional<at::Tensor>> masks) {
                auto defined_inds = _get_defined_inds(masks);
                if (defined_inds.empty())
                    return;
                for (auto iter = std::next(defined_inds.begin()); iter != defined_inds.end(); iter++)
                    check_masks_match<criteria>(masks[0], masks[*iter]);
            }

            namespace detail {
                template<TensorMatchingCriteria criteria>
                struct check_masks_match_wrapper {
                    const c10::optional<at::Tensor> mask;
                };

                template<TensorMatchingCriteria criteria>
                C10_ALWAYS_INLINE check_masks_match_wrapper<criteria> operator&(
                        const check_masks_match_wrapper<criteria> &lhs,
                        const check_masks_match_wrapper<criteria> &rhs) {
                    check_masks_match<criteria>(lhs.mask, rhs.mask);
                    return {lhs.mask};
                }
            }

            template<TensorMatchingCriteria criteria = BROADCASTABLE, typename... mask_Ts>
            C10_ALWAYS_INLINE void check_masks_match(mask_Ts &&... masks) {
                (... & detail::check_masks_match_wrapper<criteria>{masks});
            }

/// \private
///
/// Join binary masks with at::bitwise_and. If one of the masks is None, return
/// a copy of the other mask.
            template<bool copy = true, TensorMatchingCriteria criteria = BROADCASTABLE,
                    typename self_T, typename other_T>
            C10_ALWAYS_INLINE c10::optional<at::Tensor> all_masks(
                    c10::optional<self_T> self_mask,
                    c10::optional<other_T> other_mask) {
                if constexpr (std::is_same_v<self_T, at::Tensor> && std::is_same_v<other_T, at::Tensor>) {
                    if constexpr (copy) {
                        if (!self_mask.has_value())
                            return other_mask.has_value() ? other_mask->clone() : c10::optional<at::Tensor>{};
                        if (!other_mask.has_value())
                            return self_mask.has_value() ? self_mask->clone() : c10::optional<at::Tensor>{};
                    } else {
                        if (!self_mask.has_value())
                            return other_mask.has_value() ? other_mask : c10::optional<at::Tensor>{};
                        if (!other_mask.has_value())
                            return self_mask.has_value() ? self_mask : c10::optional<at::Tensor>{};
                    }
                    check_masks_match<criteria>(self_mask, other_mask);
                    return at::bitwise_and(self_mask.value(), other_mask.value());
                } else if constexpr (std::is_same_v<self_T, at::Tensor> && !std::is_same_v<other_T, at::Tensor>) {
                    if constexpr (copy) {
                        return self_mask.has_value() ? self_mask->clone() : c10::optional<at::Tensor>{};
                    } else {
                        return self_mask.has_value() ? self_mask : c10::optional<at::Tensor>{};
                    }
                } else if constexpr (!std::is_same_v<self_T, at::Tensor> && std::is_same_v<other_T, at::Tensor>) {
                    if constexpr (copy) {
                        return other_mask.has_value() ? other_mask->clone() : c10::optional<at::Tensor>{};
                    } else {
                        return other_mask.has_value() ? other_mask : c10::optional<at::Tensor>{};
                    }
                } else {
                    return {};
                }
            }

/// \private
///
/// Inplace variant of all_masks using at::Tensor::bitwise_and_ on self_mask.
/// It is returned after modification if available. Otherwise, if other_mask is
/// available, return a copy of it.
            template<bool copy = true, TensorMatchingCriteria criteria = BROADCASTABLE,
                    typename self_T, typename other_T>
            C10_ALWAYS_INLINE c10::optional<at::Tensor> all_masks_(
                    c10::optional<self_T> self_mask,
                    c10::optional<other_T> other_mask) {
                if constexpr (std::is_same_v<self_T, at::Tensor> && std::is_same_v<other_T, at::Tensor>) {
                    if constexpr (copy) {
                        if (!self_mask.has_value())
                            return other_mask.has_value() ? other_mask->clone() : c10::optional<at::Tensor>{};
                    } else {
                        if (!self_mask.has_value())
                            return other_mask.has_value() ? other_mask : c10::optional<at::Tensor>{};
                    }
                    if (!other_mask.has_value())
                        return self_mask;
                    check_masks_match<criteria>(self_mask, other_mask);
                    self_mask.value().bitwise_and_(other_mask.value());
                    return self_mask;
                } else if constexpr (std::is_same_v<self_T, at::Tensor> && !std::is_same_v<other_T, at::Tensor>) {
                    return self_mask;
                } else if constexpr (!std::is_same_v<self_T, at::Tensor> && std::is_same_v<other_T, at::Tensor>) {
                    if constexpr (copy) {
                        return other_mask.has_value() ? other_mask->clone() : c10::optional<at::Tensor>{};
                    } else {
                        return other_mask.has_value() ? other_mask : c10::optional<at::Tensor>{};
                    }
                } else {
                    return {};
                }
            }

/// \private
///
/// Join binary masks with at::bitwise_or. If one of the masks is None, return
/// a copy of the other mask.
            template<bool copy = true, TensorMatchingCriteria criteria = BROADCASTABLE,
                    typename self_T, typename other_T>
            C10_ALWAYS_INLINE c10::optional<at::Tensor> any_masks(
                    c10::optional<self_T> self_mask,
                    c10::optional<other_T> other_mask) {
                if constexpr (std::is_same_v<self_T, at::Tensor> && std::is_same_v<other_T, at::Tensor>) {
                    if constexpr (copy) {
                        if (!self_mask.has_value())
                            return other_mask.has_value() ? other_mask->clone() : c10::optional<at::Tensor>{};
                        if (!other_mask.has_value())
                            return self_mask.has_value() ? self_mask->clone() : c10::optional<at::Tensor>{};
                    } else {
                        if (!self_mask.has_value())
                            return other_mask.has_value() ? other_mask : c10::optional<at::Tensor>{};
                        if (!other_mask.has_value())
                            return self_mask.has_value() ? self_mask : c10::optional<at::Tensor>{};
                    }
                    check_masks_match<criteria>(self_mask, other_mask);
                    return at::bitwise_or(self_mask.value(), other_mask.value());
                } else {
                    return {};
                }
            }

/// \private
///
/// Inplace variant of all_masks using at::Tensor::bitwise_or_ on self_mask.
/// It is returned after modification if available. Otherwise, if other_mask is
/// available, return a copy of it.
            template<bool copy = true, TensorMatchingCriteria criteria = BROADCASTABLE,
                    typename self_T, typename other_T>
            C10_ALWAYS_INLINE c10::optional<at::Tensor> any_masks_(
                    c10::optional<self_T> self_mask,
                    c10::optional<other_T> other_mask) {
                if constexpr (std::is_same_v<self_T, at::Tensor> && std::is_same_v<other_T, at::Tensor>) {
                    if constexpr (copy) {
                        if (!self_mask.has_value())
                            return other_mask.has_value() ? other_mask->clone() : c10::optional<at::Tensor>{};
                    } else {
                        if (!self_mask.has_value())
                            return other_mask.has_value() ? other_mask : c10::optional<at::Tensor>{};
                    }
                    if (!other_mask.has_value())
                        return self_mask;
                    check_masks_match<criteria>(self_mask, other_mask);
                    self_mask.value().bitwise_or_(other_mask.value());
                    return self_mask;
                } else {
                    return {};
                }
            }

//        template<bool copy = true, typename self_T, typename... other_Ts>
//        C10_ALWAYS_INLINE c10::optional<at::Tensor> all_masks(
//                c10::optional<self_T> self_mask,
//                other_Ts &&... other_masks) {
//            return all_masks(self_mask, all_masks(other_masks...));
//        }

            namespace detail {
                template<typename T, bool commutative = false>
                struct all_masks_wrapper {
                    const T mask;
                    const bool copied;
                };

                template<bool commutative, typename T, typename U>
                C10_ALWAYS_INLINE all_masks_wrapper<c10::optional<at::Tensor>, commutative> operator&(
                        const all_masks_wrapper<T, commutative> &lhs, const all_masks_wrapper<U, commutative> &rhs) {
                    if constexpr (std::is_same_v<T, c10::optional<at::Tensor>> &&
                                  std::is_same_v<U, c10::optional<at::Tensor>>) {
                        if (lhs.copied)
                            return {all_masks_<true>(lhs.mask, rhs.mask), true};
                        if constexpr (commutative)
                            if (rhs.copied)
                                return {all_masks_<true>(rhs.mask, lhs.mask), true};
                        return {all_masks<false>(lhs.mask, rhs.mask),
                                lhs.mask.has_value() && rhs.mask.has_value()};
                    } else {
                        if (lhs.copied)
                            return {all_masks_<true>(lhs.mask, rhs.mask), true};
                        if constexpr (commutative)
                            if (rhs.copied)
                                return {all_masks_<true>(rhs.mask, lhs.mask), true};
                        return {all_masks<false>(lhs.mask, rhs.mask), lhs.copied || rhs.copied};
                    }
                }
            }

/// \private
///
/// n-ary variant of all_masks.
            template<bool copy = true, bool commutative = false, typename... mask_Ts>
            C10_ALWAYS_INLINE c10::optional<at::Tensor> all_masks(mask_Ts &&... masks) {
                auto result = (... & detail::all_masks_wrapper<mask_Ts, commutative>{masks, false});
                if constexpr (copy) {
                    if (!result.copied && result.mask.has_value())
                        return result.mask->clone();
                    else
                        return result.mask;
                } else {
                    return result.mask;
                }
            }

/// \private
///
/// n-ary variant of all_masks_.
            template<bool copy = true, bool commutative = false, typename self_T, typename... other_Ts>
            C10_ALWAYS_INLINE c10::optional<at::Tensor> all_masks_(
                    c10::optional<self_T> self_mask,
                    other_Ts &&... other_masks) {
                return all_masks_<copy>(self_mask, all_masks<false, commutative>(other_masks...));
            }

            namespace detail {
                template<typename T, bool commutative = false>
                struct any_masks_wrapper {
                    const T mask;
                    const bool copied;
                };

                template<bool commutative, typename T, typename U>
                C10_ALWAYS_INLINE any_masks_wrapper<c10::optional<at::Tensor>, commutative> operator&(
                        const any_masks_wrapper<T, commutative> &lhs, const any_masks_wrapper<U, commutative> &rhs) {
                    if constexpr (std::is_same_v<T, c10::optional<at::Tensor>> &&
                                  std::is_same_v<U, c10::optional<at::Tensor>>) {
                        if (lhs.copied)
                            return {any_masks_<true>(lhs.mask, rhs.mask), true};
                        if constexpr (commutative)
                            if (rhs.copied)
                                return {any_masks_<true>(rhs.mask, lhs.mask), true};
                        return {any_masks<false>(lhs.mask, rhs.mask),
                                lhs.mask.has_value() && rhs.mask.has_value()};
                    } else {
                        if (lhs.copied)
                            return {any_masks_<true>(lhs.mask, rhs.mask), true};
                        if constexpr (commutative)
                            if (rhs.copied)
                                return {any_masks_<true>(rhs.mask, lhs.mask), true};
                        return {any_masks<false>(lhs.mask, rhs.mask), lhs.copied || rhs.copied};
                    }
                }
            }

/// \private
///
/// n-ary variant of any_masks.
            template<bool copy = true, bool commutative = false, typename... mask_Ts>
            C10_ALWAYS_INLINE c10::optional<at::Tensor> any_masks(mask_Ts &&... masks) {
                auto result = (... & detail::any_masks_wrapper<mask_Ts, commutative>{masks, false});
                if constexpr (copy) {
                    if (!result.copied && result.mask.has_value())
                        return result.mask->clone();
                    else
                        return result.mask;
                } else {
                    return result.mask;
                }
            }

/// \private
///
/// n-ary variant of any_masks_.
            template<bool copy = true, bool commutative = false, typename self_T, typename... other_Ts>
            C10_ALWAYS_INLINE c10::optional<at::Tensor> any_masks_(
                    c10::optional<self_T> self_mask,
                    other_Ts &&... other_masks) {
                return any_masks_<copy>(self_mask, any_masks<false, commutative>(other_masks...));
            }

/// \private
///
/// Wrap a collection of data and a collection of masks into a vector of TensorMaskedPair.
            C10_ALWAYS_INLINE std::vector<c10::intrusive_ptr<TensorMaskedPair>> wrap_masked_pair(
                    at::ArrayRef<at::Tensor> data,
                    at::ArrayRef<c10::optional<at::Tensor>> masks) {
                TORCH_CHECK(data.size() == masks.size(),
                            "data and masks must have same size.")
                std::vector<c10::intrusive_ptr<TensorMaskedPair>> pairs;
                pairs.reserve(data.size());
                for (const auto i: c10::irange(data.size())) {
                    pairs.emplace_back(masked_pair(data.at(i), masks.at(i)));
                }
                return pairs;
            }

            C10_ALWAYS_INLINE std::vector<c10::intrusive_ptr<TensorMaskedPair>> wrap_masked_pair(
                    at::ArrayRef<at::Tensor> data,
                    at::ArrayRef<at::Tensor> masks) {
                TORCH_CHECK(data.size() == masks.size(),
                            "data and masks must have same size.")
                std::vector<c10::intrusive_ptr<TensorMaskedPair>> pairs;
                pairs.reserve(data.size());
                for (const auto i: c10::irange(data.size())) {
                    pairs.emplace_back(masked_pair(data.at(i), masks.at(i)));
                }
                return pairs;
            }

            C10_ALWAYS_INLINE std::vector<c10::intrusive_ptr<TensorMaskedPair>> wrap_masked_pair(
                    at::ArrayRef<at::Tensor> data,
                    const c10::optional<at::Tensor> &mask) {
                std::vector<c10::intrusive_ptr<TensorMaskedPair>> pairs;
                pairs.reserve(data.size());
                for (const auto i: c10::irange(data.size())) {
                    pairs.emplace_back(masked_pair(data.at(i), mask));
                }
                return pairs;
            }

            namespace detail {
                template<typename data_T, typename mask_T, std::size_t... Is>
                C10_ALWAYS_INLINE auto wrap_masked_pair_transform(
                        data_T &&data,
                        mask_T &&mask,
                        std::index_sequence<Is...>) {
                    if constexpr (sizeof...(Is) == 0) return std::tuple<>(); // empty tuple
                    else if constexpr (std::is_same_v<decltype(masked_pair(std::get<0>(data),
                                                                           std::get<0>(mask))), void>) {
                        (masked_pair(std::get<Is>(data), std::get<Is>(mask)), ...); // no return value expected
                        return;
                    }  // then dispatch lvalue, rvalue ref, temporary
                    else if constexpr (std::is_lvalue_reference_v<decltype(masked_pair(std::get<0>(data),
                                                                                       std::get<0>(mask)))>) {
                        return std::tie(masked_pair(std::get<Is>(data), std::get<Is>(mask))...);
                    } else if constexpr (std::is_rvalue_reference_v<decltype(masked_pair(std::get<0>(data),
                                                                                         std::get<0>(mask)))>) {
                        return std::forward_as_tuple(masked_pair(std::get<Is>(data), std::get<Is>(mask))...);
                    } else {
                        return std::tuple(masked_pair(std::get<Is>(data), std::get<Is>(mask))...);
                    }
                }
            }

/// \private
///
/// Wrap a tuple of data and a tuple of masks into a tuple of TensorMaskedPair.
            template<typename... data_Ts, typename... mask_Ts>
            C10_ALWAYS_INLINE auto wrap_masked_pair(std::tuple<data_Ts...> data,
                                                    std::tuple<mask_Ts...> masks) {
                static_assert(std::tuple_size_v<decltype(data)> == std::tuple_size_v<decltype(masks)>,
                              "data and masks must have same size.");
                return detail::wrap_masked_pair_transform(data, masks,
                                                          std::make_index_sequence<sizeof...(data_Ts)>());
            }
        }
    }
}
