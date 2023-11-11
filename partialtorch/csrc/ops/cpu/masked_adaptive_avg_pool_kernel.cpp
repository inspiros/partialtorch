#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "../utils/dispatch.h"

namespace partialtorch {
    namespace ops {
        namespace {
            namespace impl {
                template<typename index_t>
                __forceinline__ index_t start_index(index_t a, index_t b, index_t c) {
                    return (index_t) std::floor((float) (a * c) / b);
                }

                template<typename index_t>
                __forceinline__ index_t end_index(index_t a, index_t b, index_t c) {
                    return (index_t) std::ceil((float) ((a + 1) * c) / b);
                }

                template<typename scalar_t, typename index_t>
                void _masked_adaptive_avg_pool1d_forward_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> input_data,
                        const at::TensorAccessor<bool, 3> input_mask,
                        const index_t width,
                        const index_t out_w,
                        const index_t channels,
                        at::TensorAccessor<scalar_t, 3> output_data,
                        at::TensorAccessor<bool, 3> output_mask) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t w = index % out_w;
                        index_t c = (index / out_w) % channels;
                        index_t b = index / (out_w * channels);

                        scalar_t sum = 0, count = 0;
                        index_t x_start = start_index(w, out_w, width), x_end = end_index(w, out_w, width);
                        for (index_t x = x_start; x < x_end; ++x) {
                            if (input_mask[b][c][x]) {
                                sum += input_data[b][c][x];
                                count += 1;
                            }
                        }
                        output_data[b][c][w] = count ? sum / count : static_cast<scalar_t>(0);
                        output_mask[b][c][w] = count;
                    }
                }

                template<typename scalar_t, typename index_t>
                void _masked_adaptive_avg_pool2d_forward_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 4> input_data,
                        const at::TensorAccessor<bool, 4> input_mask,
                        const index_t height,
                        const index_t width,
                        const index_t out_h,
                        const index_t out_w,
                        const index_t channels,
                        at::TensorAccessor<scalar_t, 4> output_data,
                        at::TensorAccessor<bool, 4> output_mask) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t w = index % out_w;
                        index_t h = (index / out_w) % out_h;
                        index_t c = (index / (out_w * out_h)) % channels;
                        index_t b = index / (out_w * out_h * channels);

                        scalar_t sum = 0, count = 0;
                        index_t y_start = start_index(h, out_h, height), y_end = end_index(h, out_h, height);
                        index_t x_start = start_index(w, out_w, width), x_end = end_index(w, out_w, width);
                        for (index_t y = y_start; y < y_end; ++y) {
                            for (index_t x = x_start; x < x_end; ++x) {
                                if (input_mask[b][c][y][x]) {
                                    sum += input_data[b][c][y][x];
                                    count += 1;
                                }
                            }
                        }
                        output_data[b][c][h][w] = count ? sum / count : static_cast<scalar_t>(0);
                        output_mask[b][c][h][w] = count;
                    }
                }

                template<typename scalar_t, typename index_t>
                void _masked_adaptive_avg_pool3d_forward_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 5> input_data,
                        const at::TensorAccessor<bool, 5> input_mask,
                        const index_t depth,
                        const index_t height,
                        const index_t width,
                        const index_t out_d,
                        const index_t out_h,
                        const index_t out_w,
                        const index_t channels,
                        at::TensorAccessor<scalar_t, 5> output_data,
                        at::TensorAccessor<bool, 5> output_mask) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t w = index % out_w;
                        index_t h = (index / out_w) % out_h;
                        index_t d = (index / (out_w * out_h)) % out_d;
                        index_t c = (index / (out_w * out_h * out_d)) % channels;
                        index_t b = index / (out_w * out_h * out_d * channels);

                        scalar_t sum = 0, count = 0;
                        index_t z_start = start_index(d, out_d, depth), z_end = end_index(d, out_d, depth);
                        index_t y_start = start_index(h, out_h, height), y_end = end_index(h, out_h, height);
                        index_t x_start = start_index(w, out_w, width), x_end = end_index(w, out_w, width);
                        for (index_t z = z_start; z < z_end; ++z) {
                            for (index_t y = y_start; y < y_end; ++y) {
                                for (index_t x = x_start; x < x_end; ++x) {
                                    if (input_mask[b][c][z][y][x]) {
                                        sum += input_data[b][c][z][y][x];
                                        count += 1;
                                    }
                                }
                            }
                        }
                        output_data[b][c][d][h][w] = count ? sum / count : static_cast<scalar_t>(0);
                        output_mask[b][c][d][h][w] = count;
                    }
                }
            } // namespace impl

            std::tuple<at::Tensor, at::Tensor> _masked_adaptive_avg_pool1d_forward_kernel(
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size) {
                bool unbatched = data.ndimension() == 2;

                auto data_c = data.contiguous();
                auto mask_c = mask.contiguous();
                if (unbatched) {
                    data_c = data_c.unsqueeze(0);
                    mask_c = mask_c.unsqueeze(0);
                }

                int64_t batch_size = data_c.size(0);
                int64_t channels = data_c.size(1);
                int64_t in_w = data_c.size(2);

                int64_t out_w = output_size[0];

                auto output_data = at::empty({batch_size, channels, out_w}, data_c.options());
                auto output_mask = at::empty({batch_size, channels, out_w}, mask_c.options());
                int64_t n_kernels = output_data.numel();

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        data_c.scalar_type(), "_masked_adaptive_avg_pool1d_forward_cpu", ([&] {
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto output_data_accessor =
                                output_data.accessor<scalar_t, 3>();
                        auto output_mask_accessor =
                                output_mask.accessor<bool, 3>();
                        impl::_masked_adaptive_avg_pool1d_forward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                data_c.accessor<scalar_t, 3>(),
                                mask_c.accessor<bool, 3>(),
                                in_w,
                                out_w,
                                channels,
                                output_data_accessor,
                                output_mask_accessor);
                    }));
                }));
                if (unbatched) {
                    output_data.squeeze_(0);
                    output_mask.squeeze_(0);
                }
                return std::make_tuple(output_data, output_mask);
            }

            std::tuple<at::Tensor, at::Tensor> _masked_adaptive_avg_pool2d_forward_kernel(
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size) {
                bool unbatched = data.ndimension() == 3;

                auto data_c = data.contiguous();
                auto mask_c = mask.contiguous();
                if (unbatched) {
                    data_c = data_c.unsqueeze(0);
                    mask_c = mask_c.unsqueeze(0);
                }

                int64_t batch_size = data_c.size(0);
                int64_t channels = data_c.size(1);
                int64_t in_h = data_c.size(2);
                int64_t in_w = data_c.size(3);

                int64_t out_h = output_size[0];
                int64_t out_w = output_size[1];

                auto output_data = at::empty({batch_size, channels, out_h, out_w}, data_c.options());
                auto output_mask = at::empty({batch_size, channels, out_h, out_w}, mask_c.options());
                int64_t n_kernels = output_data.numel();

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        data_c.scalar_type(), "_masked_adaptive_avg_pool2d_forward_cpu", ([&] {
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto output_data_accessor =
                                output_data.accessor<scalar_t, 4>();
                        auto output_mask_accessor =
                                output_mask.accessor<bool, 4>();
                        impl::_masked_adaptive_avg_pool2d_forward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                data_c.accessor<scalar_t, 4>(),
                                mask_c.accessor<bool, 4>(),
                                in_h,
                                in_w,
                                out_h,
                                out_w,
                                channels,
                                output_data_accessor,
                                output_mask_accessor);
                    }));
                }));
                if (unbatched) {
                    output_data.squeeze_(0);
                    output_mask.squeeze_(0);
                }
                return std::make_tuple(output_data, output_mask);
            }

            std::tuple<at::Tensor, at::Tensor> _masked_adaptive_avg_pool3d_forward_kernel(
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size) {
                bool unbatched = data.ndimension() == 4;

                auto data_c = data.contiguous();
                auto mask_c = mask.contiguous();
                if (unbatched) {
                    data_c = data_c.unsqueeze(0);
                    mask_c = mask_c.unsqueeze(0);
                }

                int64_t batch_size = data_c.size(0);
                int64_t channels = data_c.size(1);
                int64_t in_d = data_c.size(2);
                int64_t in_h = data_c.size(3);
                int64_t in_w = data_c.size(4);

                int64_t out_d = output_size[0];
                int64_t out_h = output_size[1];
                int64_t out_w = output_size[2];

                auto output_data = at::empty({batch_size, channels, out_d, out_h, out_w}, data_c.options());
                auto output_mask = at::empty({batch_size, channels, out_d, out_h, out_w}, mask_c.options());
                int64_t n_kernels = output_data.numel();

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        data_c.scalar_type(), "_masked_adaptive_avg_pool3d_forward_cpu", ([&] {
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto output_data_accessor =
                                output_data.accessor<scalar_t, 5>();
                        auto output_mask_accessor =
                                output_mask.accessor<bool, 5>();
                        impl::_masked_adaptive_avg_pool3d_forward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                data_c.accessor<scalar_t, 5>(),
                                mask_c.accessor<bool, 5>(),
                                in_d,
                                in_h,
                                in_w,
                                out_d,
                                out_h,
                                out_w,
                                channels,
                                output_data_accessor,
                                output_mask_accessor);
                    }));
                }));
                if (unbatched) {
                    output_data.squeeze_(0);
                    output_mask.squeeze_(0);
                }
                return std::make_tuple(output_data, output_mask);
            }

            namespace impl {
                template<typename scalar_t, typename index_t>
                void _masked_adaptive_avg_pool1d_forward_count_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<bool, 3> input_mask,
                        const index_t width,
                        const index_t out_w,
                        const index_t channels,
                        at::TensorAccessor<scalar_t, 3> count) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t w = index % out_w;
                        index_t c = (index / out_w) % channels;
                        index_t b = index / (out_w * channels);

                        scalar_t count_val = 0;
                        index_t x_start = start_index(w, out_w, width), x_end = end_index(w, out_w, width);
                        for (index_t x = x_start; x < x_end; ++x) {
                            count_val += input_mask[b][c][x];
                        }
                        count[b][c][w] = count_val;
                    }
                }

                template<typename scalar_t, typename index_t>
                void _masked_adaptive_avg_pool2d_forward_count_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<bool, 4> input_mask,
                        const index_t height,
                        const index_t width,
                        const index_t out_h,
                        const index_t out_w,
                        const index_t channels,
                        at::TensorAccessor<scalar_t, 4> count) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t w = index % out_w;
                        index_t h = (index / out_w) % out_h;
                        index_t c = (index / (out_w * out_h)) % channels;
                        index_t b = index / (out_w * out_h * channels);

                        scalar_t count_val = 0;
                        index_t y_start = start_index(h, out_h, height), y_end = end_index(h, out_h, height);
                        index_t x_start = start_index(w, out_w, width), x_end = end_index(w, out_w, width);
                        for (index_t y = y_start; y < y_end; ++y) {
                            for (index_t x = x_start; x < x_end; ++x) {
                                count_val += input_mask[b][c][y][x];
                            }
                        }
                        count[b][c][h][w] = count_val;
                    }
                }

                template<typename scalar_t, typename index_t>
                void _masked_adaptive_avg_pool3d_forward_count_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<bool, 5> input_mask,
                        const index_t depth,
                        const index_t height,
                        const index_t width,
                        const index_t out_d,
                        const index_t out_h,
                        const index_t out_w,
                        const index_t channels,
                        at::TensorAccessor<scalar_t, 5> count) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t w = index % out_w;
                        index_t h = (index / out_w) % out_h;
                        index_t d = (index / (out_w * out_h)) % out_d;
                        index_t c = (index / (out_w * out_h * out_d)) % channels;
                        index_t b = index / (out_w * out_h * out_d * channels);

                        scalar_t count_val = 0;
                        index_t z_start = start_index(d, out_d, depth), z_end = end_index(d, out_d, depth);
                        index_t y_start = start_index(h, out_h, height), y_end = end_index(h, out_h, height);
                        index_t x_start = start_index(w, out_w, width), x_end = end_index(w, out_w, width);
                        for (index_t z = z_start; z < z_end; ++z) {
                            for (index_t y = y_start; y < y_end; ++y) {
                                for (index_t x = x_start; x < x_end; ++x) {
                                    count_val += input_mask[b][c][z][y][x];
                                }
                            }
                        }
                        count[b][c][d][h][w] = count_val;
                    }
                }

                template<typename scalar_t, typename index_t>
                void _masked_adaptive_avg_pool1d_backward_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<bool, 3> input_mask,
                        const at::TensorAccessor<scalar_t, 3> count,
                        const index_t width,
                        const index_t out_w,
                        const index_t channels,
                        at::TensorAccessor<scalar_t, 3> grad_data) {
                    CPU_1D_KERNEL_LOOP(index, n_kernels) {
                        index_t w = index % out_w;
                        index_t c = (index / out_w) % channels;
                        index_t b = index / (out_w * channels);

                        if (count[b][c][w]) {
                            index_t x_start = start_index(w, out_w, width), x_end = end_index(w, out_w, width);
                            for (index_t x = x_start; x < x_end; ++x) {
                                if (input_mask[b][c][x])
                                    grad_data[b][c][x] +=
                                            grad_output[b][c][w] / count[b][c][w];
                            }
                        }
                    }
                }

                template<typename scalar_t, typename index_t>
                void _masked_adaptive_avg_pool2d_backward_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 4> grad_output,
                        const at::TensorAccessor<bool, 4> input_mask,
                        const at::TensorAccessor<scalar_t, 4> count,
                        const index_t height,
                        const index_t width,
                        const index_t out_h,
                        const index_t out_w,
                        const index_t channels,
                        at::TensorAccessor<scalar_t, 4> grad_data) {
                    CPU_1D_KERNEL_LOOP(index, n_kernels) {
                        index_t w = index % out_w;
                        index_t h = (index / out_w) % out_h;
                        index_t c = (index / (out_w * out_h)) % channels;
                        index_t b = index / (out_w * out_h * channels);

                        if (count[b][c][h][w]) {
                            index_t y_start = start_index(h, out_h, height), y_end = end_index(h, out_h, height);
                            index_t x_start = start_index(w, out_w, width), x_end = end_index(w, out_w, width);
                            for (index_t y = y_start; y < y_end; ++y) {
                                for (index_t x = x_start; x < x_end; ++x) {
                                    if (input_mask[b][c][y][x])
                                        grad_data[b][c][y][x] +=
                                                grad_output[b][c][h][w] / count[b][c][h][w];
                                }
                            }
                        }
                    }
                }

                template<typename scalar_t, typename index_t>
                void _masked_adaptive_avg_pool3d_backward_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 5> grad_output,
                        const at::TensorAccessor<bool, 5> input_mask,
                        const at::TensorAccessor<scalar_t, 5> count,
                        const index_t depth,
                        const index_t height,
                        const index_t width,
                        const index_t out_d,
                        const index_t out_h,
                        const index_t out_w,
                        const index_t channels,
                        at::TensorAccessor<scalar_t, 5> grad_data) {
                    CPU_1D_KERNEL_LOOP(index, n_kernels) {
                        index_t w = index % out_w;
                        index_t h = (index / out_w) % out_h;
                        index_t d = (index / (out_w * out_h)) % out_d;
                        index_t c = (index / (out_w * out_h * out_d)) % channels;
                        index_t b = index / (out_w * out_h * out_d * channels);

                        if (count[b][c][d][h][w]) {
                            index_t z_start = start_index(d, out_d, depth), z_end = end_index(d, out_d, depth);
                            index_t y_start = start_index(h, out_h, height), y_end = end_index(h, out_h, height);
                            index_t x_start = start_index(w, out_w, width), x_end = end_index(w, out_w, width);
                            for (index_t z = z_start; z < z_end; ++z) {
                                for (index_t y = y_start; y < y_end; ++y) {
                                    for (index_t x = x_start; x < x_end; ++x) {
                                        if (input_mask[b][c][z][y][x])
                                            grad_data[b][c][z][y][x] +=
                                                    grad_output[b][c][d][h][w] / count[b][c][d][h][w];
                                    }
                                }
                            }
                        }
                    }
                }
            } // namespace impl

            at::Tensor _masked_adaptive_avg_pool1d_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size) {
                bool unbatched = data.ndimension() == 2;

                auto grad_output_c = grad_output.contiguous();
                auto data_c = data.contiguous();
                auto mask_c = mask.contiguous();
                if (unbatched) {
                    grad_output_c = grad_output_c.unsqueeze(0);
                    data_c = data_c.unsqueeze(0);
                    mask_c = mask_c.unsqueeze(0);
                }

                int64_t batch_size = data_c.size(0);
                int64_t channels = data_c.size(1);
                int64_t in_w = data_c.size(2);

                int64_t out_w = output_size[0];

                auto grad_data = at::zeros_like(data_c);
                auto count = at::empty_like(grad_output_c);
                int64_t n_kernels;

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        grad_output_c.scalar_type(), "_masked_adaptive_avg_pool1d_backward_cpu", ([&] {
                    n_kernels = count.numel();
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto count_accessor =
                                count.accessor<scalar_t, 3>();
                        impl::_masked_adaptive_avg_pool1d_forward_count_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                mask_c.accessor<bool, 3>(),
                                in_w,
                                out_w,
                                channels,
                                count_accessor);
                    }));

                    n_kernels = grad_output_c.numel();
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_data_accessor =
                                grad_data.accessor<scalar_t, 3>();
                        impl::_masked_adaptive_avg_pool1d_backward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                mask_c.accessor<bool, 3>(),
                                count.accessor<scalar_t, 3>(),
                                in_w,
                                out_w,
                                channels,
                                grad_data_accessor);
                    }));
                }));
                if (unbatched)
                    grad_data.squeeze_(0);
                return grad_data;
            }

            at::Tensor _masked_adaptive_avg_pool2d_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size) {
                bool unbatched = data.ndimension() == 3;

                auto grad_output_c = grad_output.contiguous();
                auto data_c = data.contiguous();
                auto mask_c = mask.contiguous();
                if (unbatched) {
                    grad_output_c = grad_output_c.unsqueeze(0);
                    data_c = data_c.unsqueeze(0);
                    mask_c = mask_c.unsqueeze(0);
                }

                int64_t batch_size = data_c.size(0);
                int64_t channels = data_c.size(1);
                int64_t in_h = data_c.size(2);
                int64_t in_w = data_c.size(3);

                int64_t out_h = output_size[0];
                int64_t out_w = output_size[1];

                auto grad_data = at::zeros_like(data_c);
                auto count = at::empty_like(grad_output_c);
                int64_t n_kernels;

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        grad_output_c.scalar_type(), "_masked_adaptive_avg_pool2d_backward_cpu", ([&] {
                    n_kernels = count.numel();
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto count_accessor =
                                count.accessor<scalar_t, 4>();
                        impl::_masked_adaptive_avg_pool2d_forward_count_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                mask_c.accessor<bool, 4>(),
                                in_h,
                                in_w,
                                out_h,
                                out_w,
                                channels,
                                count_accessor);
                    }));

                    n_kernels = grad_output_c.numel();
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_data_accessor =
                                grad_data.accessor<scalar_t, 4>();
                        impl::_masked_adaptive_avg_pool2d_backward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 4>(),
                                mask_c.accessor<bool, 4>(),
                                count.accessor<scalar_t, 4>(),
                                in_h,
                                in_w,
                                out_h,
                                out_w,
                                channels,
                                grad_data_accessor);
                    }));
                }));
                if (unbatched)
                    grad_data.squeeze_(0);
                return grad_data;
            }

            at::Tensor _masked_adaptive_avg_pool3d_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size) {
                bool unbatched = data.ndimension() == 4;

                auto grad_output_c = grad_output.contiguous();
                auto data_c = data.contiguous();
                auto mask_c = mask.contiguous();
                if (unbatched) {
                    grad_output_c = grad_output_c.unsqueeze(0);
                    data_c = data_c.unsqueeze(0);
                    mask_c = mask_c.unsqueeze(0);
                }

                int64_t batch_size = data_c.size(0);
                int64_t channels = data_c.size(1);
                int64_t in_d = data_c.size(2);
                int64_t in_h = data_c.size(3);
                int64_t in_w = data_c.size(4);

                int64_t out_d = output_size[0];
                int64_t out_h = output_size[1];
                int64_t out_w = output_size[2];

                auto grad_data = at::zeros_like(data_c);
                auto count = at::empty_like(grad_output_c);
                int64_t n_kernels;

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        grad_output_c.scalar_type(), "_masked_adaptive_avg_pool3d_backward_cpu", ([&] {
                    n_kernels = count.numel();
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto count_accessor =
                                count.accessor<scalar_t, 5>();
                        impl::_masked_adaptive_avg_pool3d_forward_count_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                mask_c.accessor<bool, 5>(),
                                in_d,
                                in_h,
                                in_w,
                                out_d,
                                out_h,
                                out_w,
                                channels,
                                count_accessor);
                    }));

                    n_kernels = grad_output_c.numel();
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_data_accessor =
                                grad_data.accessor<scalar_t, 5>();
                        impl::_masked_adaptive_avg_pool3d_backward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 5>(),
                                mask_c.accessor<bool, 5>(),
                                count.accessor<scalar_t, 5>(),
                                in_d,
                                in_h,
                                in_w,
                                out_d,
                                out_h,
                                out_w,
                                channels,
                                grad_data_accessor);
                    }));
                }

                ));
                if (unbatched)
                    grad_data.squeeze_(0);
                return
                        grad_data;
            }
        }

        TORCH_LIBRARY_IMPL(partialtorch, CPU, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::_masked_adaptive_avg_pool1d"),
                    TORCH_FN(_masked_adaptive_avg_pool1d_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::_masked_adaptive_avg_pool2d"),
                    TORCH_FN(_masked_adaptive_avg_pool2d_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::_masked_adaptive_avg_pool3d"),
                    TORCH_FN(_masked_adaptive_avg_pool3d_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::__masked_adaptive_avg_pool1d_backward"),
                    TORCH_FN(_masked_adaptive_avg_pool1d_backward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::__masked_adaptive_avg_pool2d_backward"),
                    TORCH_FN(_masked_adaptive_avg_pool2d_backward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::__masked_adaptive_avg_pool3d_backward"),
                    TORCH_FN(_masked_adaptive_avg_pool3d_backward_kernel));
        }
    }
}
