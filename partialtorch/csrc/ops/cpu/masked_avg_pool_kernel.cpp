#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "../utils/dispatch.h"

namespace partialtorch {
    namespace ops {
        namespace {
            inline int64_t compute_output_shape(int64_t input_shape,
                                                int64_t kernel_size,
                                                int64_t stride,
                                                int64_t padding,
                                                bool ceil_mode) {
                double output_shape = static_cast<double>(input_shape + 2 * padding - kernel_size) / stride + 1;
                return ceil_mode ? std::ceil(output_shape) : std::floor(output_shape);
            }

            namespace impl {
                template<bool count_include_pad, typename scalar_t, typename index_t>
                void _masked_avg_pool1d_forward_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> input_data,
                        const at::TensorAccessor<bool, 3> input_mask,
                        const index_t width,
                        const index_t kernel_w,
                        const index_t stride_w,
                        const index_t pad_w,
                        const index_t out_w,
                        const index_t channels,
                        at::TensorAccessor<scalar_t, 3> output_data,
                        at::TensorAccessor<bool, 3> output_mask) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t w = index % out_w;
                        index_t c = (index / out_w) % channels;
                        index_t b = index / (out_w * channels);

                        scalar_t sum = 0, count = 0;
                        bool mask_val = false;
                        for (index_t i = 0; i < kernel_w; ++i) {
                            const index_t x = (w * stride_w - pad_w) + i;
                            bool valid = (0 <= x && x < width) && input_mask[b][c][x];
                            sum += valid ? input_data[b][c][x] : static_cast<scalar_t>(0);
                            if constexpr (count_include_pad) {
                                bool pad_valid = !(0 <= x && x < width) || input_mask[b][c][x];
                                count += pad_valid;
                                mask_val |= pad_valid;
                            } else {
                                count += valid;
                                mask_val |= valid;
                            }
                        }
                        output_data[b][c][w] = mask_val ? (sum / count) : static_cast<scalar_t>(0);
                        output_mask[b][c][w] = mask_val;
                    }
                }

                template<bool count_include_pad, typename scalar_t, typename index_t>
                void _masked_avg_pool2d_forward_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 4> input_data,
                        const at::TensorAccessor<bool, 4> input_mask,
                        const index_t height,
                        const index_t width,
                        const index_t kernel_h,
                        const index_t kernel_w,
                        const index_t stride_h,
                        const index_t stride_w,
                        const index_t pad_h,
                        const index_t pad_w,
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
                        bool mask_val = false;
                        for (index_t i = 0; i < kernel_h; ++i) {
                            for (index_t j = 0; j < kernel_w; ++j) {
                                const index_t y = (h * stride_h - pad_h) + i;
                                const index_t x = (w * stride_w - pad_w) + j;
                                bool valid = (0 <= x && x < width &&
                                              0 <= y && y < height) && input_mask[b][c][y][x];
                                sum += valid ? input_data[b][c][y][x] : static_cast<scalar_t>(0);
                                if constexpr (count_include_pad) {
                                    bool pad_valid = !(0 <= x && x < width &&
                                                       0 <= y && y < height) || input_mask[b][c][y][x];
                                    count += pad_valid;
                                    mask_val |= pad_valid;
                                } else {
                                    count += valid;
                                    mask_val |= valid;
                                }
                            }
                        }
                        output_data[b][c][h][w] = mask_val ? (sum / count) : static_cast<scalar_t>(0);
                        output_mask[b][c][h][w] = mask_val;
                    }
                }

                template<bool count_include_pad, typename scalar_t, typename index_t>
                void _masked_avg_pool3d_forward_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 5> input_data,
                        const at::TensorAccessor<bool, 5> input_mask,
                        const index_t depth,
                        const index_t height,
                        const index_t width,
                        const index_t kernel_d,
                        const index_t kernel_h,
                        const index_t kernel_w,
                        const index_t stride_d,
                        const index_t stride_h,
                        const index_t stride_w,
                        const index_t pad_d,
                        const index_t pad_h,
                        const index_t pad_w,
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
                        bool mask_val = false;
                        for (index_t i = 0; i < kernel_d; ++i) {
                            for (index_t j = 0; j < kernel_h; ++j) {
                                for (index_t k = 0; k < kernel_w; ++k) {
                                    const index_t z = (d * stride_d - pad_d) + i;
                                    const index_t y = (h * stride_h - pad_h) + j;
                                    const index_t x = (w * stride_w - pad_w) + k;
                                    bool valid = (0 <= x && x < width &&
                                                  0 <= y && y < height &&
                                                  0 <= z && z < depth) && input_mask[b][c][z][y][x];
                                    sum += valid ? input_data[b][c][z][y][x] : static_cast<scalar_t>(0);
                                    if constexpr (count_include_pad) {
                                        bool pad_valid = !(0 <= x && x < width &&
                                                           0 <= y && y < height &&
                                                           0 <= z && z < depth) || input_mask[b][c][z][y][x];
                                        count += pad_valid;
                                        mask_val |= pad_valid;
                                    } else {
                                        count += valid;
                                        mask_val |= valid;
                                    }
                                }
                            }
                        }
                        output_data[b][c][d][h][w] = mask_val ? (sum / count) : static_cast<scalar_t>(0);
                        output_mask[b][c][d][h][w] = mask_val;
                    }
                }
            } // namespace impl

            std::tuple<at::Tensor, at::Tensor> _masked_avg_pool1d_forward_kernel(
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad) {
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

                int64_t kernel_w = kernel_size[0];

                int64_t stride_w = stride[0];

                int64_t pad_w = padding[0];

                int64_t out_w = compute_output_shape(in_w, kernel_w, stride_w, pad_w, ceil_mode);

                auto output_data = at::empty({batch_size, channels, out_w}, data_c.options());
                auto output_mask = at::empty({batch_size, channels, out_w}, mask_c.options());
                int64_t n_kernels = output_data.numel();

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        data_c.scalar_type(), "_masked_avg_pool1d_forward_cpu", ([&] {
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        PT_DISPATCH_BOOL(count_include_pad, ([&] {
                            auto output_data_accessor =
                                    output_data.accessor<scalar_t, 3>();
                            auto output_mask_accessor =
                                    output_mask.accessor<bool, 3>();
                            impl::_masked_avg_pool1d_forward_kernel_impl<count_include_pad, scalar_t, index_t>(
                                    n_kernels,
                                    data_c.accessor<scalar_t, 3>(),
                                    mask_c.accessor<bool, 3>(),
                                    in_w,
                                    kernel_w,
                                    stride_w,
                                    pad_w,
                                    out_w,
                                    channels,
                                    output_data_accessor,
                                    output_mask_accessor);
                        }));
                    }));
                }));
                if (unbatched) {
                    output_data.squeeze_(0);
                    output_mask.squeeze_(0);
                }
                return std::make_tuple(output_data, output_mask);
            }

            std::tuple<at::Tensor, at::Tensor> _masked_avg_pool2d_forward_kernel(
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad) {
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

                int64_t kernel_h = kernel_size[0];
                int64_t kernel_w = kernel_size[1];

                int64_t stride_h = stride[0];
                int64_t stride_w = stride[1];

                int64_t pad_h = padding[0];
                int64_t pad_w = padding[1];

                int64_t out_h = compute_output_shape(in_h, kernel_h, stride_h, pad_h, ceil_mode);
                int64_t out_w = compute_output_shape(in_w, kernel_w, stride_w, pad_w, ceil_mode);

                auto output_data = at::empty({batch_size, channels, out_h, out_w}, data_c.options());
                auto output_mask = at::empty({batch_size, channels, out_h, out_w}, mask_c.options());
                int64_t n_kernels = output_data.numel();

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        data_c.scalar_type(), "_masked_avg_pool2d_forward_cpu", ([&] {
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        PT_DISPATCH_BOOL(count_include_pad, ([&] {
                            auto output_data_accessor =
                                    output_data.accessor<scalar_t, 4>();
                            auto output_mask_accessor =
                                    output_mask.accessor<bool, 4>();
                            impl::_masked_avg_pool2d_forward_kernel_impl<count_include_pad, scalar_t, index_t>(
                                    n_kernels,
                                    data_c.accessor<scalar_t, 4>(),
                                    mask_c.accessor<bool, 4>(),
                                    in_h,
                                    in_w,
                                    kernel_h,
                                    kernel_w,
                                    stride_h,
                                    stride_w,
                                    pad_h,
                                    pad_w,
                                    out_h,
                                    out_w,
                                    channels,
                                    output_data_accessor,
                                    output_mask_accessor);
                        }));
                    }));
                }));
                if (unbatched) {
                    output_data.squeeze_(0);
                    output_mask.squeeze_(0);
                }
                return std::make_tuple(output_data, output_mask);
            }

            std::tuple<at::Tensor, at::Tensor> _masked_avg_pool3d_forward_kernel(
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad) {
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

                int64_t kernel_d = kernel_size[0];
                int64_t kernel_h = kernel_size[1];
                int64_t kernel_w = kernel_size[2];

                int64_t stride_d = stride[0];
                int64_t stride_h = stride[1];
                int64_t stride_w = stride[2];

                int64_t pad_d = padding[0];
                int64_t pad_h = padding[1];
                int64_t pad_w = padding[2];

                int64_t out_d = compute_output_shape(in_d, kernel_d, stride_d, pad_d, ceil_mode);
                int64_t out_h = compute_output_shape(in_h, kernel_h, stride_h, pad_h, ceil_mode);
                int64_t out_w = compute_output_shape(in_w, kernel_w, stride_w, pad_w, ceil_mode);

                auto output_data = at::empty({batch_size, channels, out_d, out_h, out_w}, data_c.options());
                auto output_mask = at::empty({batch_size, channels, out_d, out_h, out_w}, mask_c.options());
                int64_t n_kernels = output_data.numel();

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        data_c.scalar_type(), "_masked_avg_pool3d_forward_cpu", ([&] {
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        PT_DISPATCH_BOOL(count_include_pad, ([&] {
                            auto output_data_accessor =
                                    output_data.accessor<scalar_t, 5>();
                            auto output_mask_accessor =
                                    output_mask.accessor<bool, 5>();
                            impl::_masked_avg_pool3d_forward_kernel_impl<count_include_pad, scalar_t, index_t>(
                                    n_kernels,
                                    data_c.accessor<scalar_t, 5>(),
                                    mask_c.accessor<bool, 5>(),
                                    in_d,
                                    in_h,
                                    in_w,
                                    kernel_d,
                                    kernel_h,
                                    kernel_w,
                                    stride_d,
                                    stride_h,
                                    stride_w,
                                    pad_d,
                                    pad_h,
                                    pad_w,
                                    out_d,
                                    out_h,
                                    out_w,
                                    channels,
                                    output_data_accessor,
                                    output_mask_accessor);
                        }));
                    }));
                }));
                if (unbatched) {
                    output_data.squeeze_(0);
                    output_mask.squeeze_(0);
                }
                return std::make_tuple(output_data, output_mask);
            }

            namespace impl {
                template<bool count_include_pad, typename scalar_t, typename index_t>
                void _masked_avg_pool1d_forward_count_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<bool, 3> input_mask,
                        const index_t width,
                        const index_t kernel_w,
                        const index_t stride_w,
                        const index_t pad_w,
                        const index_t out_w,
                        const index_t channels,
                        at::TensorAccessor<scalar_t, 3> count) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t w = index % out_w;
                        index_t c = (index / out_w) % channels;
                        index_t b = index / (out_w * channels);

                        scalar_t count_val = 0;
                        for (index_t i = 0; i < kernel_w; ++i) {
                            const index_t x = (w * stride_w - pad_w) + i;
                            if constexpr (count_include_pad)
                                count_val += !(0 <= x && x < width) || input_mask[b][c][x];
                            else
                                count_val += (0 <= x && x < width) && input_mask[b][c][x];
                        }
                        count[b][c][w] = count_val;
                    }
                }

                template<bool count_include_pad, typename scalar_t, typename index_t>
                void _masked_avg_pool2d_forward_count_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<bool, 4> input_mask,
                        const index_t height,
                        const index_t width,
                        const index_t kernel_h,
                        const index_t kernel_w,
                        const index_t stride_h,
                        const index_t stride_w,
                        const index_t pad_h,
                        const index_t pad_w,
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
                        for (index_t i = 0; i < kernel_h; ++i) {
                            for (index_t j = 0; j < kernel_w; ++j) {
                                const index_t y = (h * stride_h - pad_h) + i;
                                const index_t x = (w * stride_w - pad_w) + j;
                                if constexpr (count_include_pad)
                                    count_val += !(0 <= x && x < width &&
                                                   0 <= y && y < height) || input_mask[b][c][y][x];
                                else
                                    count_val += (0 <= x && x < width &&
                                                  0 <= y && y < height) && input_mask[b][c][y][x];
                            }
                        }
                        count[b][c][h][w] = count_val;
                    }
                }

                template<bool count_include_pad, typename scalar_t, typename index_t>
                void _masked_avg_pool3d_forward_count_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<bool, 5> input_mask,
                        const index_t depth,
                        const index_t height,
                        const index_t width,
                        const index_t kernel_d,
                        const index_t kernel_h,
                        const index_t kernel_w,
                        const index_t stride_d,
                        const index_t stride_h,
                        const index_t stride_w,
                        const index_t pad_d,
                        const index_t pad_h,
                        const index_t pad_w,
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
                        for (index_t i = 0; i < kernel_d; ++i) {
                            for (index_t j = 0; j < kernel_h; ++j) {
                                for (index_t k = 0; k < kernel_w; ++k) {
                                    const index_t z = (d * stride_d - pad_d) + i;
                                    const index_t y = (h * stride_h - pad_h) + j;
                                    const index_t x = (w * stride_w - pad_w) + k;
                                    if constexpr (count_include_pad)
                                        count_val += !(0 <= x && x < width &&
                                                       0 <= y && y < height &&
                                                       0 <= z && z < depth) || input_mask[b][c][z][y][x];
                                    else
                                        count_val += (0 <= x && x < width &&
                                                      0 <= y && y < height &&
                                                      0 <= z && z < depth) && input_mask[b][c][z][y][x];
                                }
                            }
                        }
                        count[b][c][d][h][w] = count_val;
                    }
                }

                template<typename scalar_t, typename index_t>
                void _masked_avg_pool1d_backward_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<bool, 3> input_mask,
                        const at::TensorAccessor<scalar_t, 3> count,
                        const index_t width,
                        const index_t kernel_w,
                        const index_t stride_w,
                        const index_t pad_w,
                        const index_t out_w,
                        const index_t channels,
                        at::TensorAccessor<scalar_t, 3> grad_data) {
                    CPU_1D_KERNEL_LOOP(index, n_kernels) {
                        index_t w = index % out_w;
                        index_t c = (index / out_w) % channels;
                        index_t b = index / (out_w * channels);

                        if (count[b][c][w]) {
                            for (index_t i = 0; i < kernel_w; ++i) {
                                const index_t x = (w * stride_w - pad_w) + i;
                                bool valid = (0 <= x && x < width) && input_mask[b][c][x];
                                if (valid)
                                    grad_data[b][c][x] += grad_output[b][c][w] / count[b][c][w];
                            }
                        }
                    }
                }

                template<typename scalar_t, typename index_t>
                void _masked_avg_pool2d_backward_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 4> grad_output,
                        const at::TensorAccessor<bool, 4> input_mask,
                        const at::TensorAccessor<scalar_t, 4> count,
                        const index_t height,
                        const index_t width,
                        const index_t kernel_h,
                        const index_t kernel_w,
                        const index_t stride_h,
                        const index_t stride_w,
                        const index_t pad_h,
                        const index_t pad_w,
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
                            for (index_t i = 0; i < kernel_h; ++i) {
                                for (index_t j = 0; j < kernel_w; ++j) {
                                    const index_t y = (h * stride_h - pad_h) + i;
                                    const index_t x = (w * stride_w - pad_w) + j;
                                    bool valid = (0 <= x && x < width &&
                                                  0 <= y && y < height) && input_mask[b][c][y][x];
                                    if (valid)
                                        grad_data[b][c][y][x] += grad_output[b][c][h][w] / count[b][c][h][w];
                                }
                            }
                        }
                    }
                }

                template<typename scalar_t, typename index_t>
                void _masked_avg_pool3d_backward_kernel_impl(
                        const index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 5> grad_output,
                        const at::TensorAccessor<bool, 5> input_mask,
                        const at::TensorAccessor<scalar_t, 5> count,
                        const index_t depth,
                        const index_t height,
                        const index_t width,
                        const index_t kernel_d,
                        const index_t kernel_h,
                        const index_t kernel_w,
                        const index_t stride_d,
                        const index_t stride_h,
                        const index_t stride_w,
                        const index_t pad_d,
                        const index_t pad_h,
                        const index_t pad_w,
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
                            for (index_t i = 0; i < kernel_d; ++i) {
                                for (index_t j = 0; j < kernel_h; ++j) {
                                    for (index_t k = 0; k < kernel_w; ++k) {
                                        const index_t z = (d * stride_d - pad_d) + i;
                                        const index_t y = (h * stride_h - pad_h) + j;
                                        const index_t x = (w * stride_w - pad_w) + k;
                                        bool valid = (0 <= x && x < width &&
                                                      0 <= y && y < height &&
                                                      0 <= z && z < depth) && input_mask[b][c][z][y][x];
                                        if (valid)
                                            grad_data[b][c][z][y][x] +=
                                                    grad_output[b][c][d][h][w] / count[b][c][d][h][w];
                                    }
                                }
                            }
                        }
                    }
                }
            } // namespace impl

            at::Tensor _masked_avg_pool1d_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad) {
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

                int64_t kernel_w = kernel_size[0];

                int64_t stride_w = stride[0];

                int64_t pad_w = padding[0];

                int64_t out_w = compute_output_shape(in_w, kernel_w, stride_w, pad_w, ceil_mode);

                auto grad_data = at::zeros_like(data_c);
                auto count = at::empty_like(grad_output_c);
                int64_t n_kernels;

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        grad_output_c.scalar_type(), "_masked_avg_pool1d_backward_cpu", ([&] {
                    n_kernels = count.numel();
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        PT_DISPATCH_BOOL(count_include_pad, ([&] {
                            auto count_accessor =
                                    count.accessor<scalar_t, 3>();
                            impl::_masked_avg_pool1d_forward_count_kernel_impl<count_include_pad, scalar_t, index_t>(
                                    n_kernels,
                                    mask_c.accessor<bool, 3>(),
                                    in_w,
                                    kernel_w,
                                    stride_w,
                                    pad_w,
                                    out_w,
                                    channels,
                                    count_accessor);
                        }));
                    }));

                    n_kernels = grad_output_c.numel();
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_data_accessor =
                                grad_data.accessor<scalar_t, 3>();
                        impl::_masked_avg_pool1d_backward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                mask_c.accessor<bool, 3>(),
                                count.accessor<scalar_t, 3>(),
                                in_w,
                                kernel_w,
                                stride_w,
                                pad_w,
                                out_w,
                                channels,
                                grad_data_accessor);
                    }));
                }));
                if (unbatched)
                    grad_data.squeeze_(0);
                return grad_data;
            }

            at::Tensor _masked_avg_pool2d_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad) {
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

                int64_t kernel_h = kernel_size[0];
                int64_t kernel_w = kernel_size[1];

                int64_t stride_h = stride[0];
                int64_t stride_w = stride[1];

                int64_t pad_h = padding[0];
                int64_t pad_w = padding[1];

                int64_t out_h = compute_output_shape(in_h, kernel_h, stride_h, pad_h, ceil_mode);
                int64_t out_w = compute_output_shape(in_w, kernel_w, stride_w, pad_w, ceil_mode);

                auto grad_data = at::zeros_like(data_c);
                auto count = at::empty_like(grad_output_c);
                int64_t n_kernels;

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        grad_output_c.scalar_type(), "_masked_avg_pool2d_backward_cpu", ([&] {
                    n_kernels = count.numel();
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        PT_DISPATCH_BOOL(count_include_pad, ([&] {
                            auto count_accessor =
                                    count.accessor<scalar_t, 4>();
                            impl::_masked_avg_pool2d_forward_count_kernel_impl<count_include_pad, scalar_t, index_t>(
                                    n_kernels,
                                    mask_c.accessor<bool, 4>(),
                                    in_h,
                                    in_w,
                                    kernel_h,
                                    kernel_w,
                                    stride_h,
                                    stride_w,
                                    pad_h,
                                    pad_w,
                                    out_h,
                                    out_w,
                                    channels,
                                    count_accessor);
                        }));
                    }));

                    n_kernels = grad_output_c.numel();
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_data_accessor =
                                grad_data.accessor<scalar_t, 4>();
                        impl::_masked_avg_pool2d_backward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 4>(),
                                mask_c.accessor<bool, 4>(),
                                count.accessor<scalar_t, 4>(),
                                in_h,
                                in_w,
                                kernel_h,
                                kernel_w,
                                stride_h,
                                stride_w,
                                pad_h,
                                pad_w,
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

            at::Tensor _masked_avg_pool3d_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad) {
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

                int64_t kernel_d = kernel_size[0];
                int64_t kernel_h = kernel_size[1];
                int64_t kernel_w = kernel_size[2];

                int64_t stride_d = stride[0];
                int64_t stride_h = stride[1];
                int64_t stride_w = stride[2];

                int64_t pad_d = padding[0];
                int64_t pad_h = padding[1];
                int64_t pad_w = padding[2];

                int64_t out_d = compute_output_shape(in_d, kernel_d, stride_d, pad_d, ceil_mode);
                int64_t out_h = compute_output_shape(in_h, kernel_h, stride_h, pad_h, ceil_mode);
                int64_t out_w = compute_output_shape(in_w, kernel_w, stride_w, pad_w, ceil_mode);

                auto grad_data = at::zeros_like(data_c);
                auto count = at::empty_like(grad_output_c);
                int64_t n_kernels;

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        grad_output_c.scalar_type(), "_masked_avg_pool3d_backward_cpu", ([&] {
                    n_kernels = count.numel();
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        PT_DISPATCH_BOOL(count_include_pad, ([&] {
                            auto count_accessor =
                                    count.accessor<scalar_t, 5>();
                            impl::_masked_avg_pool3d_forward_count_kernel_impl<count_include_pad, scalar_t, index_t>(
                                    n_kernels,
                                    mask_c.accessor<bool, 5>(),
                                    in_d,
                                    in_h,
                                    in_w,
                                    kernel_d,
                                    kernel_h,
                                    kernel_w,
                                    stride_d,
                                    stride_h,
                                    stride_w,
                                    pad_d,
                                    pad_h,
                                    pad_w,
                                    out_d,
                                    out_h,
                                    out_w,
                                    channels,
                                    count_accessor);
                        }));
                    }));

                    n_kernels = grad_output_c.numel();
                    PT_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_data_accessor =
                                grad_data.accessor<scalar_t, 5>();
                        impl::_masked_avg_pool3d_backward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 5>(),
                                mask_c.accessor<bool, 5>(),
                                count.accessor<scalar_t, 5>(),
                                in_d,
                                in_h,
                                in_w,
                                kernel_d,
                                kernel_h,
                                kernel_w,
                                stride_d,
                                stride_h,
                                stride_w,
                                pad_d,
                                pad_h,
                                pad_w,
                                out_d,
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
        }

        TORCH_LIBRARY_IMPL(partialtorch, CPU, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::_masked_avg_pool1d"),
                    TORCH_FN(_masked_avg_pool1d_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::_masked_avg_pool2d"),
                    TORCH_FN(_masked_avg_pool2d_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::_masked_avg_pool3d"),
                    TORCH_FN(_masked_avg_pool3d_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::__masked_avg_pool1d_backward"),
                    TORCH_FN(_masked_avg_pool1d_backward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::__masked_avg_pool2d_backward"),
                    TORCH_FN(_masked_avg_pool2d_backward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::__masked_avg_pool3d_backward"),
                    TORCH_FN(_masked_avg_pool3d_backward_kernel));
        }
    }
}
