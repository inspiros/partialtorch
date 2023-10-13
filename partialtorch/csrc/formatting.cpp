#include <iostream>

#include <ATen/core/Formatting.h>

#include "MaskedPair.h"

namespace partialtorch {
    namespace {
        const std::string MASKED_VALUE = "--";

        //not all C++ compilers have default float so we define our own here
        inline std::ios_base &defaultfloat(std::ios_base &__base) {
            __base.unsetf(std::ios_base::floatfield);
            return __base;
        }

        //saves/restores number formatting inside scope
        struct FormatGuard {
            FormatGuard(std::ostream &out)
                    : out(out), saved(nullptr) {
                saved.copyfmt(out);
            }

            ~FormatGuard() {
                out.copyfmt(saved);
            }

        private:
            std::ostream &out;
            std::ios saved;
        };

        std::tuple<double, int64_t> __printFormat(std::ostream &stream, const at::Tensor &self) {
            auto size = self.numel();
            if (size == 0) {
                return std::make_tuple(1., 0);
            }
            bool intMode = true;
            auto self_p = self.data_ptr<double>();
            for (const auto i: c10::irange(size)) {
                auto z = self_p[i];
                if (std::isfinite(z)) {
                    if (z != std::ceil(z)) {
                        intMode = false;
                        break;
                    }
                }
            }
            int64_t offset = 0;
            while (!std::isfinite(self_p[offset])) {
                offset = offset + 1;
                if (offset == size) {
                    break;
                }
            }
            double expMin = 1;
            double expMax = 1;
            if (offset != size) {
                expMin = fabs(self_p[offset]);
                expMax = fabs(self_p[offset]);
                for (const auto i: c10::irange(offset, size)) {
                    double z = fabs(self_p[i]);
                    if (std::isfinite(z)) {
                        if (z < expMin) {
                            expMin = z;
                        }
                        if (self_p[i] > expMax) {
                            expMax = z;
                        }
                    }
                }
                if (expMin != 0) {
                    expMin = std::floor(std::log10(expMin)) + 1;
                } else {
                    expMin = 1;
                }
                if (expMax != 0) {
                    expMax = std::floor(std::log10(expMax)) + 1;
                } else {
                    expMax = 1;
                }
            }
            double scale = 1;
            int64_t sz = 11;
            if (intMode) {
                if (expMax > 9) {
                    sz = 11;
                    stream << std::scientific << std::setprecision(4);
                } else {
                    sz = expMax + 1;
                    stream << defaultfloat;
                }
            } else {
                if (expMax - expMin > 4) {
                    sz = 11;
                    if (std::fabs(expMax) > 99 || std::fabs(expMin) > 99) {
                        sz = sz + 1;
                    }
                    stream << std::scientific << std::setprecision(4);
                } else {
                    if (expMax > 5 || expMax < 0) {
                        sz = 7;
                        scale = std::pow(10, expMax - 1);
                        stream << std::fixed << std::setprecision(4);
                    } else {
                        if (expMax == 0) {
                            sz = 7;
                        } else {
                            sz = expMax + 6;
                        }
                        stream << std::fixed << std::setprecision(4);
                    }
                }
            }
            return std::make_tuple(scale, sz);
        }

        void __printIndent(std::ostream &stream, int64_t indent) {
            for (C10_UNUSED const auto i: c10::irange(indent)) {
                stream << " ";
            }
        }

        void printScale(std::ostream &stream, double scale) {
            FormatGuard guard(stream);
            stream << defaultfloat << scale << " *" << std::endl;
        }

        void __printMatrix(std::ostream &stream,
                           const at::Tensor &self,
                           const at::Tensor &mask,
                           int64_t linesize,
                           int64_t indent) {
            double scale = 0.0;
            int64_t sz = 0;
            std::tie(scale, sz) = __printFormat(stream, self);

            __printIndent(stream, indent);
            int64_t nColumnPerLine = (linesize - indent) / (sz + 1);
            int64_t firstColumn = 0;
            int64_t lastColumn = -1;
            while (firstColumn < self.size(1)) {
                if (firstColumn + nColumnPerLine <= self.size(1)) {
                    lastColumn = firstColumn + nColumnPerLine - 1;
                } else {
                    lastColumn = self.size(1) - 1;
                }
                if (nColumnPerLine < self.size(1)) {
                    if (firstColumn != 0) {
                        stream << std::endl;
                    }
                    stream << "Columns " << firstColumn + 1 << " to " << lastColumn + 1;
                    __printIndent(stream, indent);
                }
                if (scale != 1) {
                    printScale(stream, scale);
                    __printIndent(stream, indent);
                }
                for (const auto l: c10::irange(self.size(0))) {
                    at::Tensor row = self.select(0, l);
                    at::Tensor row_mask = mask.select(0, l);
                    double *row_ptr = row.data_ptr<double>();
                    bool *row_mask_ptr = row_mask.data_ptr<bool>();
                    for (const auto c: c10::irange(firstColumn, lastColumn + 1)) {
                        stream << std::setw(sz);
                        if (row_mask_ptr[c])
                            stream << row_ptr[c] / scale;
                        else
                            stream << MASKED_VALUE;
                        if (c == lastColumn) {
                            stream << std::endl;
                            if (l != self.size(0) - 1) {
                                if (scale != 1) {
                                    __printIndent(stream, indent);
                                    stream << " ";
                                } else {
                                    __printIndent(stream, indent);
                                }
                            }
                        } else {
                            stream << " ";
                        }
                    }
                }
                firstColumn = lastColumn + 1;
            }
        }

        void __printTensor(std::ostream &stream,
                           const at::Tensor &self,
                           const at::Tensor &mask,
                           int64_t linesize) {
            std::vector<int64_t> counter(self.ndimension() - 2);
            bool start = true;
            bool finished = false;
            counter[0] = -1;
            for (const auto i: c10::irange(1, counter.size())) {
                counter[i] = 0;
            }
            while (true) {
                for (int64_t i = 0; self.ndimension() - 2; i++) {
                    counter[i] = counter[i] + 1;
                    if (counter[i] >= self.size(i)) {
                        if (i == self.ndimension() - 3) {
                            finished = true;
                            break;
                        }
                        counter[i] = 0;
                    } else {
                        break;
                    }
                }
                if (finished) {
                    break;
                }
                if (start) {
                    start = false;
                } else {
                    stream << std::endl;
                }
                stream << "(";
                at::Tensor tensor = self, mask_tensor = mask;
                for (const auto i: c10::irange(self.ndimension() - 2)) {
                    tensor = tensor.select(0, counter[i]);
                    mask_tensor = mask_tensor.select(0, counter[i]);
                    stream << counter[i] + 1 << ",";
                }
                stream << ".,.) = " << std::endl;
                __printMatrix(stream, tensor, mask_tensor, linesize, 1);
            }
        }
    }

    std::ostream &print(std::ostream &stream, const MaskedPair<at::Tensor> &p, int64_t linesize) {
        if (!p.mask_.has_value()) {
            print(stream, p.data_, linesize);
            return stream;
        }
        FormatGuard guard(stream);
        auto data_ = p.data_, mask_ = p.mask_.value();
        TORCH_CHECK_VALUE(
                !p.mask_.has_value() || mask_.sizes() == data_.sizes(),
                "data_ and mask_ has incompatible sizes. Got data_.sizes()=",
                data_.sizes(),
                ", while mask_.sizes()=",
                mask_.sizes())

        if (!data_.defined()) {
            stream << "[ MaskedTensor (undefined) ]";
        } else if (data_.is_sparse()) {
            stream << "[ Masked" << data_.toString() << "{}\n";
            stream << "indices:\n" << data_._indices() << "\n";
            stream << "values:\n" << data_._values() << "\n";
            stream << "size:\n" << data_.sizes() << "\n";
            stream << "masked_indices:\n" << mask_._indices() << "\n";
            stream << "]";
        } else {
            at::Tensor data, mask;

            if (data_.is_quantized()) {
                data = data_.dequantize().to(at::kCPU, at::kDouble).contiguous();
                mask = mask_.dequantize().to(at::kCPU, at::kBool).contiguous();
            } else if (data_.is_mkldnn()) {
                stream << "MKLDNN Tensor: ";
                data = data_.to_dense().to(at::kCPU, at::kDouble).contiguous();
                mask = mask_.to_dense().to(at::kCPU, at::kBool).contiguous();
            } else if (data_.is_mps()) {
                // MPS does not support double tensors, so first copy then convert
                data = data_.to(at::kCPU, at::kDouble).contiguous();
                mask = mask_.to(at::kCPU, at::kBool).contiguous();
            } else {
                data = data_.to(at::kCPU, at::kDouble).contiguous();
                mask = mask_.to(at::kCPU, at::kBool).contiguous();
            }

            if (data.ndimension() == 0) {
                if (mask.data_ptr<bool>()[0])
                    stream << defaultfloat << data.data_ptr<double>()[0] << std::endl;
                else
                    stream << MASKED_VALUE << std::endl;
                stream << "[ Masked" << data_.toString() << "{}";
            } else if (data.ndimension() == 1) {
                if (data.numel() > 0) {
                    double scale = 0.0;
                    int64_t sz = 0;
                    std::tie(scale, sz) = __printFormat(stream, data);
                    if (scale != 1) {
                        printScale(stream, scale);
                    }
                    double *data_p = data.data_ptr<double>();
                    bool *mask_p = mask.data_ptr<bool>();
                    for (const auto i: c10::irange(data.size(0))) {
                        stream << std::setw(sz);
                        if (mask_p[i])
                            stream << data_p[i] / scale << std::endl;
                        else
                            stream << MASKED_VALUE << std::endl;
                    }
                }
                stream << "[ Masked" << data_.toString() << "{" << data.size(0) << "}";
            } else if (data.ndimension() == 2) {
                if (data.numel() > 0) {
                    __printMatrix(stream, data, mask, linesize, 0);
                }
                stream << "[ Masked" << data_.toString() << "{" << data.size(0) << "," << data.size(1) << "}";
            } else {
                if (data.numel() > 0) {
                    __printTensor(stream, data, mask, linesize);
                }
                stream << "[ Masked" << data_.toString() << "{" << data.size(0);
                for (const auto i: c10::irange(1, data.ndimension())) {
                    stream << "," << data.size(i);
                }
                stream << "}";
            }

            if (data_.is_quantized()) {
                stream << ", qscheme: " << toString(data_.qscheme());
                if (data_.qscheme() == c10::kPerTensorAffine) {
                    stream << ", scale: " << data_.q_scale();
                    stream << ", zero_point: " << data_.q_zero_point();
                } else if (data_.qscheme() == c10::kPerChannelAffine ||
                           data_.qscheme() == c10::kPerChannelAffineFloatQParams) {
                    stream << ", scales: ";
                    at::Tensor scales = data_.q_per_channel_scales();
                    print(stream, scales, linesize);
                    stream << ", zero_points: ";
                    at::Tensor zero_points = data_.q_per_channel_zero_points();
                    print(stream, zero_points, linesize);
                    stream << ", axis: " << data_.q_per_channel_axis();
                }
            }

            // Proxy check for if autograd was built
            if (data.getIntrusivePtr()->autograd_meta()) {
                auto &fw_grad = data._fw_grad(/* level */ 0);
                if (fw_grad.defined()) {
                    stream << ", tangent:" << std::endl << fw_grad;
                }
            }

            stream << " ]";
        }
        return stream;
    }

    void print(const MaskedPair<at::Tensor> &p, int64_t linesize) {
        print(std::cout, p, linesize);
    }
}
