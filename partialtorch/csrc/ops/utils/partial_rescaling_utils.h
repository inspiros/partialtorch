#pragma once

#include "mask_utils.h"

namespace partialtorch {
    namespace ops {
        namespace utils {
            template<typename T>
            C10_ALWAYS_INLINE at::Tensor ones_like(const T &input,
                                 at::TensorOptions options = {},
                                 c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {
                return at::ones_like(utils::get_data(input), options, memory_format);
            }

            template<typename T>
            C10_ALWAYS_INLINE at::Tensor ones_like(const T &input,
                                 c10::optional<at::ScalarType> dtype,
                                 c10::optional<at::Layout> layout,
                                 c10::optional<at::Device> device,
                                 c10::optional<bool> pin_memory,
                                 c10::optional<at::MemoryFormat> memory_format) {
                return at::ones_like(utils::get_data(input), dtype, layout, device, pin_memory, memory_format);
            }

            template<typename T>
            C10_ALWAYS_INLINE std::vector<at::Tensor> ones_like(at::ArrayRef<T> inputs,
                                              at::TensorOptions options = {},
                                              c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {
                std::vector<at::Tensor> outputs;
                outputs.reserve(inputs.size());
                for (const auto &input: inputs) {
                    outputs.emplace_back(ones_like(input, options, memory_format));
                }
                return outputs;
            }

            template<typename T>
            C10_ALWAYS_INLINE std::vector<at::Tensor> ones_like(at::ArrayRef<T> inputs,
                                              c10::optional<at::ScalarType> dtype,
                                              c10::optional<at::Layout> layout,
                                              c10::optional<at::Device> device,
                                              c10::optional<bool> pin_memory,
                                              c10::optional<at::MemoryFormat> memory_format) {
                std::vector<at::Tensor> outputs;
                outputs.reserve(inputs.size());
                for (const auto &input: inputs) {
                    outputs.emplace_back(ones_like(input, dtype, layout, device, pin_memory, memory_format));
                }
                return outputs;
            }
        }
    }
}