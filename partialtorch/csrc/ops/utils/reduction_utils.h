#pragma once

#include <ATen/ATen.h>

#include "fill_identity.h"

namespace partialtorch {
    namespace ops {
        namespace utils {
/// \private
///
/// Return dim argument as a tuple of sorted dim values.
            C10_ALWAYS_INLINE std::vector<int64_t> canonical_dim(
                    at::OptionalIntArrayRef dim,
                    int64_t ndim) {
                // TODO: in the future, dim = {} will translate to "no reduction".
                //  This change in behavior must be handled.
                std::vector<int64_t> dim_;
                if (!dim.has_value() || dim->empty()) {
                    dim_.reserve(ndim);
                    auto range = c10::irange(ndim);
                    std::copy(range.begin(), range.end(), std::back_inserter(dim_));
                    return dim_;
                }
                ndim = std::max(ndim, static_cast<int64_t>(1));
                dim_.reserve(dim->size());
                for (const auto d: *dim) {
                    TORCH_CHECK(std::find(dim_.begin(), dim_.end(), d) == dim_.end(),
                                "dim=", d, " appears multiple times in the list of dim")
                    TORCH_CHECK_INDEX(-ndim <= d < ndim,
                                      "Dimension out of range (expected to be in range of [",
                                      -ndim, ", ", ndim - 1, "], but got ", d, ")")
                    dim_.emplace_back(d % ndim);
                }
                std::sort(dim_.begin(), dim_.end());
                return dim_;
            }

            struct reduction_dims_grouper {
                std::vector<int64_t> dim_;
                std::vector<int64_t> grouped_shape;
                std::vector<int64_t> keepdim_shape;
                bool identity, keepdim;

                C10_ALWAYS_INLINE reduction_dims_grouper() : identity(true), keepdim(true) {}

                [[maybe_unused]] C10_ALWAYS_INLINE reduction_dims_grouper(
                        at::IntArrayRef input_dims,
                        at::IntArrayRef dim = {},
                        bool keepdim = false) : keepdim(keepdim) {
                    if (dim.empty()) {
                        identity = true;
                        return;
                    }
                    identity = false;

                    initialize_grouped_shape(input_dims, dim);
                }

                [[maybe_unused]] C10_ALWAYS_INLINE reduction_dims_grouper(
                        at::IntArrayRef input_dims,
                        const std::set<int64_t> &dim = {},
                        bool keepdim = false) : keepdim(keepdim) {
                    if (dim.empty()) {
                        identity = true;
                        return;
                    }
                    identity = false;
                    std::vector<int64_t> dim_vec;
                    dim_vec.reserve(dim.size());
                    std::copy(dim.begin(), dim.end(), dim_vec.begin());

                    initialize_grouped_shape(input_dims, dim_vec);
                }

                static C10_ALWAYS_INLINE bool check_needs_grouping(at::IntArrayRef dim) {
                    if (dim.size() < 2)
                        return false;
                    for (auto dim_iter = std::next(dim.begin()); dim_iter != dim.end(); ++dim_iter) {
                        if (*dim_iter == *std::prev(dim_iter) + 1)
                            return true;
                    }
                    return false;
                }

                C10_ALWAYS_INLINE at::Tensor group(const at::Tensor &input) const {
                    if (!identity)
                        return input.view(grouped_shape);
                    return input;
                }

                C10_ALWAYS_INLINE at::Tensor ungroup(const at::Tensor &output) const {
                    if (!identity && keepdim)
                        return output.view(keepdim_shape);
                    return output;
                }

            private:
                void initialize_grouped_shape(
                        at::IntArrayRef input_dims,
                        at::IntArrayRef dim = {}) {
                    // This method assumes dim contains only positive values and is sorted
                    dim_.reserve(dim.size());
                    grouped_shape.reserve(input_dims.size());
                    if (keepdim)
                        keepdim_shape.reserve(input_dims.size());

                    int64_t group_d = 1;
                    auto dim_iter = dim.begin();
                    for (const auto d: c10::irange(input_dims.size())) {
                        if (d < *dim_iter || dim_iter == dim.end()) {
                            if (keepdim)
                                keepdim_shape.emplace_back(input_dims[d]);
                            grouped_shape.emplace_back(input_dims[d]);
                        } else {
                            if (keepdim)
                                keepdim_shape.emplace_back(1);
                            group_d *= input_dims[d];
                            if (dim_iter == std::prev(dim.end())) {
                                grouped_shape.emplace_back(group_d);
                                dim_.emplace_back(grouped_shape.size() - 1);
                            } else if (*dim_iter + 1 == *std::next(dim_iter)) {
                                // consecutive
                            } else {
                                grouped_shape.emplace_back(group_d);
                                dim_.emplace_back(grouped_shape.size() - 1);
                                group_d = 1;
                            }
                            dim_iter++;
                        }
                    }
                }
            };

            namespace impl {
                template<typename zero_dim_op_T, typename single_dim_op_T, typename... Args>
                C10_ALWAYS_INLINE at::Tensor multidim_reduction_impl(
                        const zero_dim_op_T &zd_op,
                        const single_dim_op_T &sd_op,
                        const at::Tensor &self,
                        at::OptionalIntArrayRef dim = {},
                        bool keepdim = false,
                        Args &&... args) {
                    // TODO: in the future, dim = {} will translate to "no reduction".
                    //  This change in behavior must be handled.
                    std::vector<int64_t> dim_;
                    if (!dim.has_value() || dim->empty()) {
                        if (!keepdim)
                            return zd_op.call(self, args...);
                        else {
                            dim_.reserve(self.ndimension());
                            auto range = c10::irange(self.ndimension());
                            std::copy(range.begin(), range.end(), std::back_inserter(dim_));
                        }
                    } else {
                        dim_ = canonical_dim(dim, self.ndimension());
                    }

                    if (reduction_dims_grouper::check_needs_grouping(self.sizes())) {
                        auto g = reduction_dims_grouper(self.sizes(), dim_, keepdim);
                        auto r = g.group(self);
                        for (auto d = g.dim_.rbegin(); d != g.dim_.rend(); ++d)
                            r = sd_op.call(r, *d, keepdim, args...);
                        return g.ungroup(r);
                    } else {
                        auto r = self;
                        for (auto d = dim_.rbegin(); d != dim_.rend(); ++d)
                            r = sd_op.call(r, *d, keepdim, args...);
                        return r;
                    }
                }
            }

/// \private
///
/// Multi-dimensional all.
            C10_ALWAYS_INLINE at::Tensor all(
                    const at::Tensor &self,
                    at::OptionalIntArrayRef dim = {},
                    bool keepdim = false) {
                static const auto zd_op = at::_ops::all();
                static const auto sd_op = at::_ops::all_dim();
                return impl::multidim_reduction_impl(zd_op, sd_op, self, dim, keepdim);
            }

            C10_ALWAYS_INLINE c10::optional<at::Tensor> all(
                    const c10::optional<at::Tensor> &mask,
                    at::OptionalIntArrayRef dim = {},
                    bool keepdim = false) {
                if (!mask.has_value())
                    return {};
                return all(mask.value(), dim, keepdim);
            }

            struct all_dim_IntList {
                static C10_ALWAYS_INLINE at::Tensor call(const at::Tensor &self,
                                                         at::OptionalIntArrayRef dim,
                                                         bool keepdim) {
                    return all(self, dim, keepdim);
                }
            };

/// \private
///
/// Multi-dimensional any.
            C10_ALWAYS_INLINE at::Tensor any(
                    const at::Tensor &self,
                    at::OptionalIntArrayRef dim = {},
                    bool keepdim = false) {
                static const auto zd_op = at::_ops::any();
                static const auto sd_op = at::_ops::any_dim();
                return impl::multidim_reduction_impl(zd_op, sd_op, self, dim, keepdim);
            }

            C10_ALWAYS_INLINE c10::optional<at::Tensor> any(
                    const c10::optional<at::Tensor> &mask,
                    at::OptionalIntArrayRef dim = {},
                    bool keepdim = false) {
                if (!mask.has_value())
                    return {};
                return any(mask.value(), dim, keepdim);
            }

            struct any_dim_IntList {
                static C10_ALWAYS_INLINE at::Tensor call(const at::Tensor &self,
                                                         at::OptionalIntArrayRef dim,
                                                         bool keepdim) {
                    return any(self, dim, keepdim);
                }
            };

/// \private
///
/// Multi-dimensional prod.
            C10_ALWAYS_INLINE at::Tensor prod(
                    const at::Tensor &self,
                    at::OptionalIntArrayRef dim = {},
                    bool keepdim = false,
                    c10::optional<at::ScalarType> dtype = {}) {
                static const auto zd_op = at::_ops::prod();
                static const auto sd_op = at::_ops::prod_dim_int();
                return impl::multidim_reduction_impl(zd_op, sd_op, self, dim, keepdim, dtype);
            }

            struct prod_dim_IntList {
                static C10_ALWAYS_INLINE at::Tensor call(const at::Tensor &self,
                                                         at::OptionalIntArrayRef dim,
                                                         bool keepdim,
                                                         c10::optional<at::ScalarType> dtype) {
                    return prod(self, dim, keepdim, dtype);
                }
            };
        }
    }
}
