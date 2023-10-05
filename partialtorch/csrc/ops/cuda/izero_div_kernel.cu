#include <ATen/ATen.h>
#include <torch/library.h>

#include "cuda_helpers.h"
#include "../utils/dispatch.h"

namespace partialtorch {
    namespace ops {
        namespace utils {
            namespace {
                static constexpr auto default_result_type = at::kFloat;

                constexpr unsigned int GET_THREADS() {
                    return 1024;
                }

                namespace impl {
                    enum BinaryRestrictPtrScheme {
                        OutOfPlace, Inplace, LeftInplace
                    };

                    template<BinaryRestrictPtrScheme scheme = OutOfPlace, typename scalar_t, typename index_t>
                    __launch_bounds__(1024) __global__ void izero_div_kernel_impl(
                            index_t n_kernels,
                            std::conditional_t<scheme == Inplace,
                                    const scalar_t *, const scalar_t *__restrict__> self,
                            std::conditional_t<scheme == LeftInplace,
                                    const scalar_t *, const scalar_t *__restrict__> other,
                            std::conditional_t<scheme == OutOfPlace,
                                    scalar_t *__restrict__, scalar_t *> output) {
                        CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                            scalar_t o = other[index];
                            output[index] = o == static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) : self[index] / o;
                        }
                    }

                    template<BinaryRestrictPtrScheme scheme = OutOfPlace, typename scalar_t, typename index_t>
                    __launch_bounds__(1024) __global__ void izero_div_kernel_impl(
                            index_t n_kernels,
                            scalar_t self,
                            std::conditional_t<scheme == LeftInplace,
                                    const scalar_t *, const scalar_t *__restrict__> other,
                            std::conditional_t<scheme == OutOfPlace,
                                    scalar_t *__restrict__, scalar_t *> output) {
                        CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                            scalar_t o = other[index];
                            output[index] = o == static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) : self / o;
                        }
                    }

                    template<BinaryRestrictPtrScheme scheme = OutOfPlace, typename scalar_t, typename index_t>
                    __launch_bounds__(1024) __global__ void izero_div_kernel_impl(
                            index_t n_kernels,
                            std::conditional_t<scheme == Inplace,
                                    const scalar_t *, const scalar_t *__restrict__> self,
                            scalar_t other,
                            std::conditional_t<scheme == OutOfPlace,
                                    scalar_t *__restrict__, scalar_t *> output) {
                        if (other == static_cast<scalar_t>(0)) {
                            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                                output[index] = static_cast<scalar_t>(0);
                            }
                        } else {
                            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                                output[index] = self[index] / other;
                            }
                        }
                    }
                }  // namespace impl

                template<bool left = false>
                at::Tensor izero_div_kernel(
                        const at::Tensor &self,
                        const at::Tensor &other) {
                    at::checkAllSameGPU("izero_div_kernel", {
                            at::TensorArg(self, "self", 0),
                            at::TensorArg(other, "other", 1),
                    });
                    TORCH_CHECK(self.sizes() == other.sizes(),
                                "Broadcasting semantic is not implemented for izero_div. "
                                "Expect self's shape to be identical to other's shape. Got self.shape=",
                                self.sizes(),
                                ", while other.shape=",
                                other.sizes())

                    at::cuda::CUDAGuard device_guard(self.get_device());
                    auto result_type = at::result_type(self, other);
                    if (at::isIntegralType(result_type, true))
                        result_type = default_result_type;
                    auto self_c = self.contiguous().to(result_type);
                    auto other_c = other.contiguous().to(result_type);

                    auto output = at::empty_like(self_c);
                    int64_t n_kernels = output.numel();

                    auto self_flatten = self_c.flatten();
                    auto other_flatten = other_c.flatten();
                    auto output_flatten = output.flatten();

                    const unsigned int threads = GET_THREADS();
                    const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

                    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
                            at::kHalf, at::kBFloat16, result_type, "izero_div", ([&] {
                        PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CUDA, ([&] {
                            if constexpr (left) {
                                impl::izero_div_kernel_impl<impl::OutOfPlace, scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                other_flatten.data_ptr<scalar_t>(),
                                                self_flatten.data_ptr<scalar_t>(),
                                                output_flatten.data_ptr<scalar_t>());
                            } else {
                                impl::izero_div_kernel_impl<impl::OutOfPlace, scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                self_flatten.data_ptr<scalar_t>(),
                                                other_flatten.data_ptr<scalar_t>(),
                                                output_flatten.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    return output;
                }

                template<bool left = false>
                at::Tensor izero_div_Scalar_kernel(
                        const at::Tensor &self,
                        const at::Scalar &other) {
                    at::cuda::CUDAGuard device_guard(self.get_device());
                    auto result_type = at::result_type(self, other);
                    if (at::isIntegralType(result_type, true))
                        result_type = default_result_type;
                    auto self_c = self.contiguous().to(result_type);

                    auto output = at::empty_like(self_c);
                    int64_t n_kernels = output.numel();

                    auto self_flatten = self_c.flatten();
                    auto output_flatten = output.flatten();

                    const unsigned int threads = GET_THREADS();
                    const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

                    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
                            at::kHalf, at::kBFloat16, result_type, "izero_div_Scalar", ([&] {
                        PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CUDA, ([&] {
                            if constexpr (left) {
                                impl::izero_div_kernel_impl<impl::OutOfPlace, scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                other.to<scalar_t>(),
                                                self_flatten.data_ptr<scalar_t>(),
                                                output_flatten.data_ptr<scalar_t>());
                            } else {
                                impl::izero_div_kernel_impl<impl::OutOfPlace, scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                self_flatten.data_ptr<scalar_t>(),
                                                other.to<scalar_t>(),
                                                output_flatten.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    return output;
                }

                template<bool left = false>
                at::Tensor izero_div_rScalar_kernel(
                        const at::Scalar &self,
                        const at::Tensor &other) {
                    return izero_div_Scalar_kernel<!left>(other, self);
                }

                template<bool left = false>
                at::Tensor izero_div__kernel(
                        at::Tensor &self,
                        const at::Tensor &other) {
                    at::checkAllSameGPU("izero_div__kernel", {
                            at::TensorArg(self, "self", 0),
                            at::TensorArg(other, "other", 1),
                    });
                    TORCH_CHECK(self.sizes() == other.sizes(),
                                "Broadcasting semantic is not implemented for izero_div_. "
                                "Expect self's shape to be identical to other's shape. Got self.shape=",
                                self.sizes(),
                                ", while other.shape=",
                                other.sizes())

                    at::cuda::CUDAGuard device_guard(self.get_device());
                    auto result_type = at::result_type(self, other);
                    if (at::isIntegralType(result_type, true))
                        result_type = default_result_type;
                    TORCH_CHECK(at::can_cast(result_type, self.scalar_type()),
                                "result type ",
                                result_type,
                                " can't be cast to the desired output type ",
                                self.scalar_type())
                    auto self_c = self.contiguous().to(result_type);
                    auto other_c = other.contiguous().to(result_type);

                    bool needs_cast = !self.is_contiguous() || result_type != self.scalar_type();
                    auto output = needs_cast ? at::empty_like(self_c) : self_c;
                    int64_t n_kernels = self.numel();

                    auto output_flatten = output.flatten();
                    auto self_flatten = self_c.flatten();
                    auto other_flatten = other_c.flatten();

                    const unsigned int threads = GET_THREADS();
                    const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

                    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
                            at::kHalf, at::kBFloat16, result_type, "izero_div_", ([&] {
                        PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CUDA, ([&] {
                            if constexpr (left) {
                                impl::izero_div_kernel_impl<impl::LeftInplace, scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                other_flatten.data_ptr<scalar_t>(),
                                                self_flatten.data_ptr<scalar_t>(),
                                                output.data_ptr<scalar_t>());
                            } else {
                                impl::izero_div_kernel_impl<impl::Inplace, scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                self_flatten.data_ptr<scalar_t>(),
                                                other_flatten.data_ptr<scalar_t>(),
                                                output.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                    C10_CUDA_KERNEL_LAUNCH_CHECK();

                    if (needs_cast)
                        self.copy_(output);
                    return self;
                }

                template<bool left = false>
                at::Tensor izero_div__Scalar_kernel(
                        at::Tensor &self,
                        const at::Scalar &other) {
                    at::cuda::CUDAGuard device_guard(self.get_device());
                    auto result_type = at::result_type(self, other);
                    if (at::isIntegralType(result_type, true))
                        result_type = default_result_type;
                    TORCH_CHECK(at::can_cast(result_type, self.scalar_type()),
                                "result type ",
                                result_type,
                                " can't be cast to the desired output type ",
                                self.scalar_type())
                    auto self_c = self.contiguous().to(result_type);

                    bool needs_cast = !self.is_contiguous();
                    auto output = needs_cast ? at::empty_like(self_c) : self_c;
                    int64_t n_kernels = self.numel();

                    auto output_flatten = output.flatten();
                    auto self_flatten = self_c.flatten();

                    const unsigned int threads = GET_THREADS();
                    const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

                    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
                            at::kHalf, at::kBFloat16, result_type, "izero_div__Scalar", ([&] {
                        PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CUDA, ([&] {
                            if constexpr (left) {
                                impl::izero_div_kernel_impl<impl::LeftInplace, scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                other.to<scalar_t>(),
                                                self_flatten.data_ptr<scalar_t>(),
                                                output.data_ptr<scalar_t>());
                            } else {
                                impl::izero_div_kernel_impl<impl::Inplace, scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                self_flatten.data_ptr<scalar_t>(),
                                                other.to<scalar_t>(),
                                                output.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                    C10_CUDA_KERNEL_LAUNCH_CHECK();

                    if (needs_cast)
                        self.copy_(output);
                    return self;
                }

                namespace impl {
                    template<typename scalar_t, typename index_t>
                    __launch_bounds__(1024) __global__ void izero_div_backward_kernel_impl(
                            index_t n_kernels,
                            const scalar_t *__restrict__ grad_output,
                            const scalar_t *__restrict__ self,
                            const scalar_t *__restrict__ other,
                            scalar_t *__restrict__ grad_self,
                            scalar_t *__restrict__ grad_other) {
                        CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                            scalar_t o = other[index];
                            if (o != static_cast<scalar_t>(0)) {
                                scalar_t g_s = grad_output[index] / o;
                                grad_self[index] = g_s;
                                grad_other[index] = g_s * -self[index] / o;
                            }
                        }
                    }

                    template<typename scalar_t, typename index_t>
                    __launch_bounds__(1024) __global__ void izero_div_backward_kernel_impl(
                            index_t n_kernels,
                            const scalar_t *__restrict__ grad_output,
                            scalar_t other,
                            scalar_t *__restrict__ grad_self) {
                        if (other != static_cast<scalar_t>(0)) {
                            CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                                grad_self[index] = grad_output[index] / other;
                            }
                        }
                    }

                    template<typename scalar_t, typename index_t>
                    __launch_bounds__(1024) __global__ void izero_div_backward_kernel_impl(
                            index_t n_kernels,
                            const scalar_t *__restrict__ grad_output,
                            scalar_t self,
                            const scalar_t *__restrict__ other,
                            scalar_t *__restrict__ grad_other) {
                        CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                            scalar_t o = other[index];
                            if (o != static_cast<scalar_t>(0)) {
                                grad_other[index] = grad_output[index] * -self / (o * o);
                            }
                        }
                    }
                }  // namespace impl

                template<bool left = false>
                std::tuple<at::Tensor, at::Tensor> izero_div_backward_kernel(
                        const at::Tensor &grad_output,
                        const at::Tensor &self,
                        const at::Tensor &other) {
                    at::cuda::CUDAGuard device_guard(grad_output.get_device());
                    auto result_type = grad_output.scalar_type();
                    auto grad_output_c = grad_output.contiguous();
                    auto self_c = self.contiguous().to(result_type);
                    auto other_c = other.contiguous().to(result_type);

                    int64_t n_kernels = grad_output.numel();
                    auto grad_self = at::zeros_like(self_c);
                    auto grad_other = at::zeros_like(other_c);

                    auto grad_output_flatten = grad_output.flatten();
                    auto self_flatten = self.flatten();
                    auto other_flatten = other.flatten();
                    auto grad_self_flatten = grad_self.flatten();
                    auto grad_other_flatten = grad_other.flatten();

                    const unsigned int threads = GET_THREADS();
                    const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

                    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
                            at::kHalf, at::kBFloat16, result_type, "izero_div_backward", ([&] {
                        PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CUDA, ([&] {
                            if constexpr (left) {
                                impl::izero_div_backward_kernel_impl<scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                grad_output_flatten.data_ptr<scalar_t>(),
                                                other_flatten.data_ptr<scalar_t>(),
                                                self_flatten.data_ptr<scalar_t>(),
                                                grad_other_flatten.data_ptr<scalar_t>(),
                                                grad_self_flatten.data_ptr<scalar_t>());
                            } else {
                                impl::izero_div_backward_kernel_impl<scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                grad_output_flatten.data_ptr<scalar_t>(),
                                                self_flatten.data_ptr<scalar_t>(),
                                                other_flatten.data_ptr<scalar_t>(),
                                                grad_self_flatten.data_ptr<scalar_t>(),
                                                grad_other_flatten.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    return std::make_tuple(grad_self.to(self.scalar_type()), grad_other.to(other.scalar_type()));
                }

                template<bool left = false>
                at::Tensor izero_div_backward_Scalar_kernel(
                        const at::Tensor &grad_output,
                        const at::Tensor &self,
                        const at::Scalar &other) {
                    at::cuda::CUDAGuard device_guard(grad_output.get_device());
                    auto result_type = grad_output.scalar_type();
                    auto grad_output_c = grad_output.contiguous();
                    auto self_c = self.contiguous().to(result_type);

                    int64_t n_kernels = grad_output.numel();
                    auto grad_self = at::zeros_like(self_c);

                    auto grad_output_flatten = grad_output.flatten();
                    auto self_flatten = self.flatten();
                    auto grad_self_flatten = grad_self.flatten();

                    const unsigned int threads = GET_THREADS();
                    const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

                    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
                            at::kHalf, at::kBFloat16, result_type, "izero_div_Scalar_backward", ([&] {
                        PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CUDA, ([&] {
                            if constexpr (left) {  // TODO
                                impl::izero_div_backward_kernel_impl<scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                grad_output_flatten.data_ptr<scalar_t>(),
                                                other.to<scalar_t>(),
                                                self_flatten.data_ptr<scalar_t>(),
                                                grad_self_flatten.data_ptr<scalar_t>());
                            } else {
                                impl::izero_div_backward_kernel_impl<scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                grad_output_flatten.data_ptr<scalar_t>(),
                                                other.to<scalar_t>(),
                                                grad_self_flatten.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    return grad_self.to(self.scalar_type());
                }

                template<bool left = false>
                at::Tensor izero_div_backward_rScalar_kernel(
                        const at::Tensor &grad_output,
                        const at::Scalar &self,
                        const at::Tensor &other) {
                    at::cuda::CUDAGuard device_guard(grad_output.get_device());
                    auto result_type = grad_output.scalar_type();
                    auto grad_output_c = grad_output.contiguous();
                    auto other_c = other.contiguous().to(result_type);

                    int64_t n_kernels = grad_output.numel();
                    auto grad_other = at::zeros_like(other_c);

                    auto grad_output_flatten = grad_output.flatten();
                    auto other_flatten = other.flatten();
                    auto grad_other_flatten = grad_other.flatten();

                    const unsigned int threads = GET_THREADS();
                    const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

                    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
                            at::kHalf, at::kBFloat16, result_type, "izero_div_rScalar_backward", ([&] {
                        PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CUDA, ([&] {
                            if constexpr (left) {
                                impl::izero_div_backward_kernel_impl<scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                grad_output_flatten.data_ptr<scalar_t>(),
                                                self.to<scalar_t>(),
                                                grad_other_flatten.data_ptr<scalar_t>());
                            } else {
                                impl::izero_div_backward_kernel_impl<scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                                grad_output_flatten.data_ptr<scalar_t>(),
                                                self.to<scalar_t>(),
                                                other_flatten.data_ptr<scalar_t>(),
                                                grad_other_flatten.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    return grad_other.to(other.scalar_type());
                }
            }

            TORCH_LIBRARY_IMPL(partialtorch, CUDA, m) {
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_div"),
                        TORCH_FN(izero_div_kernel<false>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_ldiv"),
                        TORCH_FN(izero_div_kernel<true>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_div.Scalar"),
                        TORCH_FN(izero_div_Scalar_kernel<false>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_ldiv.Scalar"),
                        TORCH_FN(izero_div_Scalar_kernel<true>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_div.rScalar"),
                        TORCH_FN(izero_div_rScalar_kernel<false>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_ldiv.rScalar"),
                        TORCH_FN(izero_div_rScalar_kernel<true>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_div_"),
                        TORCH_FN(izero_div__kernel<false>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_ldiv_"),
                        TORCH_FN(izero_div__kernel<true>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_div_.Scalar"),
                        TORCH_FN(izero_div__Scalar_kernel<false>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_ldiv_.Scalar"),
                        TORCH_FN(izero_div__Scalar_kernel<true>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::__izero_div_backward"),
                        TORCH_FN(izero_div_backward_kernel<false>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::__izero_ldiv_backward"),
                        TORCH_FN(izero_div_backward_kernel<true>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::__izero_div_backward.Scalar"),
                        TORCH_FN(izero_div_backward_Scalar_kernel<false>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::__izero_ldiv_backward.Scalar"),
                        TORCH_FN(izero_div_backward_Scalar_kernel<true>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::__izero_div_backward.rScalar"),
                        TORCH_FN(izero_div_backward_rScalar_kernel<false>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::__izero_ldiv_backward.rScalar"),
                        TORCH_FN(izero_div_backward_rScalar_kernel<true>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::__izero_div__backward"),
                        TORCH_FN(izero_div_backward_kernel<false>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::__izero_ldiv__backward"),
                        TORCH_FN(izero_div_backward_kernel<true>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::__izero_div__backward.Scalar"),
                        TORCH_FN(izero_div_backward_Scalar_kernel<false>));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::__izero_ldiv__backward.Scalar"),
                        TORCH_FN(izero_div_backward_Scalar_kernel<true>));
            }
        }
    }
}
