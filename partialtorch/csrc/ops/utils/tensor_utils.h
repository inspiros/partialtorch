#pragma once

#include <ATen/ATen.h>

namespace partialtorch {
    namespace ops {
        namespace utils {
            C10_ALWAYS_INLINE bool shapes_broadcastable(
                    at::IntArrayRef self_shape,
                    at::IntArrayRef other_shape) {
                int64_t max_len = std::max(self_shape.size(), other_shape.size());
                for (auto i = -1; i > -1 - max_len; i--) {
                    int64_t i_ = self_shape.size() + i, j_ = other_shape.size() + i;
                    if (i_ < 0 || j_ < 0)
                        break;
                    if (self_shape[i_] < 0 || other_shape[j_] < 0)
                        return false;
                    if (self_shape[i_] == 1 || other_shape[j_] == 1 || self_shape[i_] == other_shape[j_])
                        continue;
                    return false;
                }
                return true;
            }

            C10_ALWAYS_INLINE bool shapes_broadcastable(
                    at::ArrayRef<at::IntArrayRef> shapes) {
                int64_t max_len = 0;
                for (const auto &shape: shapes) {
                    if (shape.size() > max_len)
                        max_len = shape.size();
                }
                for (auto i = -1; i > -1 - max_len; i--) {
                    int64_t result_shape = 1;
                    for (const auto &shape: shapes) {
                        int64_t i_ = shape.size() + i;
                        if (i_ < 0)
                            continue;
                        if (shape[i_] < 0)
                            return false;
                        if (shape[i_] == 1 || shape[i_] == result_shape)
                            continue;
                        if (result_shape != 1)
                            return false;
                        result_shape = shape[i_];
                    }
                }
                return true;
            }

            C10_ALWAYS_INLINE bool shapes_broadcastable(
                    std::initializer_list<at::IntArrayRef> shapes) {
                int64_t max_len = 0;
                for (const auto &shape: shapes) {
                    if (shape.size() > max_len)
                        max_len = shape.size();
                }
                for (auto i = -1; i > -1 - max_len; i--) {
                    int64_t result_shape = 1;
                    for (const auto &shape: shapes) {
                        int64_t i_ = shape.size() + i;
                        if (i_ < 0)
                            continue;
                        if (shape[i_] < 0)
                            return false;
                        if (shape[i_] == 1 || shape[i_] == result_shape)
                            continue;
                        if (result_shape != 1)
                            return false;
                        result_shape = shape[i_];
                    }
                }
                return true;
            }

            C10_ALWAYS_INLINE std::vector<int64_t> broadcast_shapes(
                    at::ArrayRef<at::IntArrayRef> shapes) {
                int64_t max_len = 0;
                for (const auto &shape: shapes) {
                    if (shape.size() > max_len)
                        max_len = shape.size();
                }
                std::vector<int64_t> result(max_len, 1);
                for (const auto &shape: shapes) {
                    for (auto i = shape.size() - 1; i > 0; i--) {
                        TORCH_CHECK(shape[i] >= 0,
                                    "Trying to create tensor with negative dimension (",
                                    shape[i],
                                    "): (",
                                    shape[i],
                                    ")")
                        if (shape[i] == 1 || shape[i] == result[i])
                            continue;
                        TORCH_CHECK(result[i] == 1,
                                    "Shape mismatch: objects cannot be broadcast to a single shape")
                        result[i] = shape[i];
                    }
                }
                return result;
            }

            enum TensorMatchingCriteria {
                EXACT,
                SHAPE,
                BROADCASTABLE,
            };


            template<TensorMatchingCriteria criteria = EXACT>
            C10_ALWAYS_INLINE bool tensors_match(
                    const at::Tensor &self,
                    const at::Tensor &other,
                    double rtol = 1e-05,
                    double atol = 1e-08) {
                if constexpr (criteria == EXACT)
                    return at::isclose(self, self.dtype() == other.dtype() ? other : other.type_as(self),
                                       rtol, atol).all().item<bool>();
                else if constexpr (criteria == SHAPE)
                    return self.sizes() == other.sizes();
                else if constexpr (criteria == BROADCASTABLE)
                    return shapes_broadcastable(self.sizes(), other.sizes());
            }

            template<TensorMatchingCriteria criteria = EXACT>
            C10_ALWAYS_INLINE bool tensors_match(
                    at::ArrayRef<at::Tensor> tensors,
                    double rtol = 1e-05,
                    double atol = 1e-08) {
                for (const auto i: c10::irange(1, tensors.size())) {
                    if (!tensors_match<criteria>(tensors[0], tensors[i], rtol, atol))
                        return false;
                }
                return true;
            }

            namespace detail {
                template<TensorMatchingCriteria criteria = EXACT>
                struct tensors_match_wrapper {
                    const at::Tensor data;
                    const bool match;
                };

                template<TensorMatchingCriteria criteria>
                C10_ALWAYS_INLINE tensors_match_wrapper<criteria> operator&(
                        const tensors_match_wrapper<criteria> &lhs, const tensors_match_wrapper<criteria> &rhs) {
                    if (!lhs.match || !rhs.match)
                        return {at::Tensor{}, false};
                    else
                        return {rhs.data, tensors_match<criteria>(lhs.data, rhs.data)};
                }
            }

            template<TensorMatchingCriteria criteria = EXACT, typename... Ts>
            C10_ALWAYS_INLINE bool tensors_match(Ts &&... tensors) {
                auto result = (... & detail::tensors_match_wrapper<criteria>{tensors, true});
                return result.match;
            }

            template<TensorMatchingCriteria criteria = EXACT>
            C10_ALWAYS_INLINE void check_tensors_match(
                    const at::Tensor &self,
                    const at::Tensor &other,
                    double rtol = 1e-05,
                    double atol = 1e-08) {
                TORCH_CHECK(tensors_match<criteria>(self, other, rtol, atol),
                            "Tensors do not match.")
            }

            template<TensorMatchingCriteria criteria = EXACT>
            C10_ALWAYS_INLINE void check_tensors_match(
                    at::ArrayRef<at::Tensor> tensors,
                    double rtol = 1e-05,
                    double atol = 1e-08) {
                for (const auto i: c10::irange(1, tensors.size()))
                    check_tensors_match<criteria>(tensors[0], tensors[i], rtol, atol);
            }

            namespace detail {
                template<TensorMatchingCriteria criteria = EXACT>
                struct check_tensors_match_wrapper {
                    const at::Tensor data;
                };

                template<TensorMatchingCriteria criteria>
                C10_ALWAYS_INLINE check_tensors_match_wrapper<criteria> operator&(
                        const check_tensors_match_wrapper<criteria> &lhs,
                        const check_tensors_match_wrapper<criteria> &rhs) {
                    check_tensors_match<criteria>(lhs.mask, rhs.mask);
                    return {lhs.mask};
                }
            }

            template<TensorMatchingCriteria criteria = EXACT, typename... mask_Ts>
            C10_ALWAYS_INLINE void check_tensors_match(mask_Ts &&... tensors) {
                (... & detail::check_tensors_match_wrapper<criteria>{tensors});
            }
        }
    }
}
