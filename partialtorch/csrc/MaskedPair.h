#pragma once

#include <ostream>
#include <ATen/Tensor.h>
#include <ATen/core/Formatting.h>
#include <torch/custom_class.h>

#include "utils/mask_utils.h"
#include "macros.h"

namespace partialtorch {
    template<typename T>
    struct PARTIALTORCH_API MaskedPair : public c10::intrusive_ptr_target {
    public:
        T data_;
        c10::optional<bool> mask_;

        using data_type = T;
        using mask_type = bool;

        MaskedPair(
                const T &data,
                const c10::optional<bool> mask = c10::nullopt) {
            this->data_ = data;
            this->mask_ = mask;
        }

        C10_ALWAYS_INLINE T get_data() const {
            return this->data_;
        }

        C10_ALWAYS_INLINE void set_data(const T &data) {
            this->data_ = data;
        }

        C10_ALWAYS_INLINE c10::optional<bool> get_mask() const {
            return this->mask_;
        }

        C10_ALWAYS_INLINE void set_mask(const c10::optional<bool> mask) {
            this->mask_ = mask;
        }

        C10_ALWAYS_INLINE std::tuple<T, c10::optional<bool>> get_members() const {
            return std::make_tuple(this->data_, this->mask_);
        }

        C10_ALWAYS_INLINE void set_members(std::tuple<T, c10::optional<bool>> members) {
            this->data_ = std::get<0>(members);
            this->mask_ = std::get<1>(members);
        }

        inline std::string toString() const {
            std::ostringstream ss;
            ss << this;
            return ss.str();
        }
    };

    template<>
    struct PARTIALTORCH_API MaskedPair<at::Tensor> : public torch::CustomClassHolder {
    public:
        at::Tensor data_;
        c10::optional<at::Tensor> mask_;

#define TENSORMASKEDPAIR_SCHEMA_STR "__torch__.torch.classes.partialtorch.MaskedPair"
        STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, TENSORMASKEDPAIR_SCHEMA_STR)
        using data_type = at::Tensor;
        using mask_type = at::Tensor;

        MaskedPair(
                const at::Tensor &data,
                const c10::optional<at::Tensor> &mask = c10::nullopt) {
            this->data_ = data;
            this->mask_ = mask;
            validate_members();
        }

        inline at::Tensor get_data() const {
            return data_;
        }

        inline void set_data(const at::Tensor &data) {
            data_ = data;
            validate_members();
        }

        inline void set_data_(const at::Tensor &data) {
            data_.copy_(data);
        }

        inline c10::optional<at::Tensor> get_mask() const {
            return mask_;
        }

        inline void set_mask(const c10::optional<at::Tensor> &mask) {
            mask_ = mask;
            validate_members();
        }

        inline void set_mask_(const c10::optional<at::Tensor> &mask) {
            if (mask.has_value() && mask_.has_value())
                mask_.value().copy_(mask.value());
            else
                mask_ = mask;
        }

        inline std::tuple<at::Tensor, c10::optional<at::Tensor>> get_members() const {
            return std::make_tuple(this->data_, this->mask_);
        }

        inline void set_members(std::tuple<at::Tensor, c10::optional<at::Tensor>> members) {
            this->data_ = std::get<0>(members);
            this->mask_ = std::get<1>(members);
            validate_members();
        }

        inline std::string toString() const {
            std::ostringstream ss;
            ss << this;
            return ss.str();
        }

    private:
        C10_ALWAYS_INLINE void validate_members() const {
            validate_masked_pair(this->data_, this->mask_);
        }

    public:
        // properties
        inline int64_t dim() const {
            return data_.dim();
        }

        inline int64_t storage_offset() const {
            return data_.storage_offset();
        }

        inline bool is_complex() const {
            return data_.is_complex();
        }

        inline bool is_floating_point() const {
            return data_.is_floating_point();
        }

        inline bool is_signed() const {
            return data_.is_signed();
        }

        inline c10::SymInt sym_size(int64_t dim) const {
            return data_.sym_size(dim);
        }

        inline c10::SymInt sym_stride(int64_t dim) const {
            return data_.sym_stride(dim);
        }

        inline int64_t size(int64_t dim) const {
            return data_.size(dim);
        }

        inline int64_t stride(int64_t dim) const {
            return data_.stride(dim);
        }

        inline bool defined() const {
            return data_.defined();
        }

        inline at::IntArrayRef sizes() const {
            return data_.sizes();
        }

        inline c10::SymIntArrayRef sym_sizes() const {
            return data_.sym_sizes();
        }

        inline c10::SymIntArrayRef sym_strides() const {
            return data_.sym_strides();
        }

        inline at::IntArrayRef strides() const {
            return data_.strides();
        }

        inline c10::optional<at::DimnameList> opt_names() const {
            return data_.opt_names();
        }

        inline at::DimnameList names() const {
            return data_.names();
        }

        inline int64_t ndimension() const {
            return data_.ndimension();
        }

        inline bool is_contiguous(at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const {
            return data_.is_contiguous(memory_format);
        }

        inline int64_t numel() const {
            return data_.numel();
        }

        inline c10::SymInt sym_numel() const {
            return data_.sym_numel();
        }

        inline c10::SymInt sym_storage_offset() const {
            return data_.sym_storage_offset();
        }

        inline int64_t itemsize() const {
            return data_.itemsize();
        }

        inline int64_t element_size() const {
            return data_.element_size();
        }

        inline at::ScalarType scalar_type() const {
            return data_.scalar_type();
        }

        inline bool has_storage() const {
            return data_.has_storage();
        }

        inline const at::Storage &storage() const {
            return data_.storage();
        }

        inline bool is_alias_of(const at::TensorBase &other) const {
            return data_.is_alias_of(other);
        }

        inline bool _is_zerotensor() const {
            return data_._is_zerotensor();
        }

        inline bool is_conj() const {
            return data_.is_conj();
        }

        inline bool is_neg() const {
            return data_.is_neg();
        }

        inline at::Layout layout() const {
            return data_.layout();
        }

        inline caffe2::TypeMeta dtype() const {
            return data_.dtype();
        }

        inline at::Device device() const {
            return data_.device();
        }

        inline int64_t get_device() const {
            return data_.get_device();
        }

        inline bool is_cpu() const {
            return data_.is_cpu();
        }

        inline bool is_cuda() const {
            return data_.is_cuda();
        }

        inline bool is_ipu() const {
            return data_.is_ipu();
        }

        inline bool is_xpu() const {
            return data_.is_xpu();
        }

        inline bool is_xla() const {
            return data_.is_xla();
        }

        inline bool is_hpu() const {
            return data_.is_hpu();
        }

        inline bool is_lazy() const {
            return data_.is_lazy();
        }

        inline bool is_hip() const {
            return data_.is_hip();
        }

        inline bool is_ve() const {
            return data_.is_ve();
        }

        inline bool is_sparse() const {
            return data_.is_sparse();
        }

        inline bool is_sparse_csr() const {
            return data_.is_sparse_csr();
        }

        inline bool is_mkldnn() const {
            return data_.is_mkldnn();
        }

        inline bool is_mps() const {
            return data_.is_mps();
        }

        inline bool is_ort() const {
            return data_.is_ort();
        }

        inline bool is_vulkan() const {
            return data_.is_vulkan();
        }

        inline bool is_metal() const {
            return data_.is_metal();
        }

        inline bool is_quantized() const {
            return data_.is_quantized();
        }

        inline bool is_meta() const {
            return data_.is_meta();
        }

        inline bool is_inference() const {
            return data_.is_inference();
        }

        inline bool is_nested() const {
            return data_.is_nested();
        }

        inline bool has_names() const {
            return data_.has_names();
        }

        inline at::TensorOptions options() const {
            return data_.options();
        }

        inline const c10::intrusive_ptr<MaskedPair<at::Tensor>> set_requires_grad(bool requires_grad) const {
            data_.set_requires_grad(requires_grad);
            return get_intrusive_ptr();
        }

        inline bool requires_grad() const {
            return data_.requires_grad();
        }

        inline const std::shared_ptr<torch::autograd::Node> &grad_fn() const {
            return data_.grad_fn();
        }

        inline bool is_leaf() const {
            return data_.is_leaf();
        }

        inline int64_t output_nr() const {
            return data_.output_nr();
        }

        inline int64_t _version() const {
            return data_._version();
        }

        inline void retain_grad() const {
            data_.retain_grad();
        }

        inline bool retains_grad() const {
            return data_.retains_grad();
        }

        inline const c10::intrusive_ptr<MaskedPair<at::Tensor>> requires_grad_(bool _requires_grad = true) const {
            data_.requires_grad_(_requires_grad);
            return get_intrusive_ptr();
        }

        inline bool is_view() const {
            return data_.is_view();
        }

        inline const at::TensorBase& _base() const {
            return data_._base();
        }

        inline const std::string& name() const {
            return data_.name();
        }

        inline at::Tensor &mutable_grad() const {
            return data_.mutable_grad();
        }

        inline const at::Tensor &grad() const {
            return data_.grad();
        }

        // ~~~~ ops ~~~~
        C10_ALWAYS_INLINE c10::intrusive_ptr<MaskedPair<at::Tensor>> get_intrusive_ptr() const {
            return at::make_intrusive<MaskedPair<at::Tensor>>(*this);
        }

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> cpu(
                c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const {
            return to(options().device(at::DeviceType::CPU), false, false, memory_format);
        }

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> cuda(
                at::Device device = at::DeviceType::CUDA,
                bool non_blocking = false,
                c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const {
            TORCH_CHECK(device.is_cuda(),
                        "Invalid device, must be cuda device")
            return to(options().device(device), non_blocking, false, memory_format);
        }

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> hip(
                at::Device device = at::DeviceType::HIP,
                bool non_blocking = false,
                c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const {
            TORCH_CHECK(device.is_hip(),
                        "Invalid device, must be hip device")
            return to(options().device(device), non_blocking, false, memory_format);
        }

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> ve(
                at::Device device = at::DeviceType::VE,
                bool non_blocking = false,
                c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const {
            TORCH_CHECK(device.is_ve(),
                        "Invalid device, must be ve device")
            return to(options().device(device), non_blocking, false, memory_format);
        }

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> vulkan(
                at::Device device = at::DeviceType::Vulkan,
                bool non_blocking = false,
                c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const {
            TORCH_CHECK(device.is_vulkan(),
                        "Invalid device, must be vulkan device")
            return to(options().device(device), non_blocking, false, memory_format);
        }

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> metal(
                at::Device device = at::DeviceType::Metal,
                bool non_blocking = false,
                c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const {
            TORCH_CHECK(device.is_metal(),
                        "Invalid device, must be metal device")
            return to(options().device(device), non_blocking, false, memory_format);
        }

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> meta(
                at::Device device = at::DeviceType::Meta,
                bool non_blocking = false,
                c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const {
            TORCH_CHECK(device.is_meta(),
                        "Invalid device, must be meta device")
            return to(options().device(device), non_blocking, false, memory_format);
        }

        inline void backward(
                const at::Tensor &gradient = {},
                c10::optional<bool> retain_graph = c10::nullopt,
                bool create_graph = false,
                c10::optional<at::TensorList> inputs = c10::nullopt) const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> clone(
                c10::optional<at::MemoryFormat> memory_format = at::MemoryFormat::Preserve) const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> contiguous(
                at::MemoryFormat memory_format = at::MemoryFormat::Preserve) const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> detach() const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> detach_();

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> fill_masked(const at::Scalar &value) const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> fill_masked(const at::Tensor &value) const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> fill_masked_(const at::Scalar &value);

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> fill_masked_(const at::Tensor &value);

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> to(
                at::TensorOptions options,
                bool non_blocking = false,
                bool copy = false,
                c10::optional<at::MemoryFormat> memory_format = c10::nullopt) const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> to(
                c10::optional<at::ScalarType> dtype = c10::nullopt,
                c10::optional<at::Layout> layout = c10::nullopt,
                c10::optional<at::Device> device = c10::nullopt,
                c10::optional<bool> pin_memory = c10::nullopt,
                bool non_blocking = false,
                bool copy = false,
                c10::optional<at::MemoryFormat> memory_format = c10::nullopt) const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> to(
                at::Device device,
                at::ScalarType dtype,
                bool non_blocking = false,
                bool copy = false,
                c10::optional<at::MemoryFormat> memory_format = c10::nullopt) const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> to(
                at::ScalarType dtype,
                bool non_blocking = false,
                bool copy = false,
                c10::optional<at::MemoryFormat> memory_format = c10::nullopt) const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> to(
                const c10::intrusive_ptr<MaskedPair<at::Tensor>> &other,
                bool non_blocking = false,
                bool copy = false,
                c10::optional<at::MemoryFormat> memory_format = c10::nullopt) const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> to(
                const at::Tensor &other,
                bool non_blocking = false,
                bool copy = false,
                c10::optional<at::MemoryFormat> memory_format = c10::nullopt) const;

        inline at::Tensor to_tensor(const at::Scalar &value) const;

        inline at::Scalar item(const at::Scalar &value) const {
            return mask_.has_value() && !mask_->item<bool>() ? value : data_.item();
        }

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> index(at::ArrayRef<at::indexing::TensorIndex> indices) const;

        inline at::Tensor index_non_masked() const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> view(at::SymIntArrayRef size) const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> t() const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> t_();

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> transpose(int64_t dim0, int64_t dim1) const;

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> transpose_(int64_t dim0, int64_t dim1);

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> permute(at::IntArrayRef dims) const;

        // Python API
        inline c10::optional<at::Tensor> __getitem__(int64_t item) const;

        inline c10::optional<at::Tensor> __next__();

        inline c10::intrusive_ptr<MaskedPair<at::Tensor>> __iter__();

        inline std::string __str__() const;

        inline std::string __repr__() const;

    private:
        int64_t _idx = 0;
    };

    template<typename T>
    PARTIALTORCH_API C10_ALWAYS_INLINE c10::intrusive_ptr<MaskedPair<T>> masked_pair(
            const MaskedPair<T> &pair) {
        return at::make_intrusive<MaskedPair<T>>(pair);
    }

    template<typename T>
    PARTIALTORCH_API C10_ALWAYS_INLINE const c10::intrusive_ptr<MaskedPair<T>> &masked_pair(
            const c10::intrusive_ptr<MaskedPair<T>> &pair) {
        return pair;
    }

    template<typename T,
            typename std::enable_if_t<!std::is_same_v<T, at::Tensor>, bool> = true>
    PARTIALTORCH_API C10_ALWAYS_INLINE c10::intrusive_ptr<MaskedPair<T>> masked_pair(
            const T &data,
            const c10::optional<bool> mask = c10::nullopt) {
        return at::make_intrusive<MaskedPair<T>>(MaskedPair<T>(data, mask));
    }

    PARTIALTORCH_API C10_ALWAYS_INLINE c10::intrusive_ptr<MaskedPair<at::Tensor>> masked_pair(
            const at::Tensor &data,
            const c10::optional<at::Tensor> &mask = c10::nullopt) {
        return at::make_intrusive<MaskedPair<at::Tensor>>(MaskedPair<at::Tensor>(data, mask));
    }

    PARTIALTORCH_API C10_ALWAYS_INLINE c10::intrusive_ptr<MaskedPair<at::Tensor>> masked_pair(
            const at::Tensor &data,
            const c10::optional<bool> mask) {
        if (!mask.has_value() || mask.value())
            return at::make_intrusive<MaskedPair<at::Tensor>>(MaskedPair<at::Tensor>(data, {}));
        else {
            auto tensor_mask = at::zeros_like(data, at::TensorOptions(at::kBool));
            return at::make_intrusive<MaskedPair<at::Tensor>>(MaskedPair<at::Tensor>(data, tensor_mask));
        }
    }

    template<typename T>
    PARTIALTORCH_API C10_ALWAYS_INLINE c10::intrusive_ptr<MaskedPair<T>> masked_pair(
            std::tuple<T, std::conditional_t<
                    std::is_same_v<T, at::Tensor>,
                    c10::optional<at::Tensor>,
                    c10::optional<bool>>> args) {
        return at::make_intrusive<MaskedPair<T>>(MaskedPair<T>(std::get<0>(args), std::get<1>(args)));
    }

    template<typename T>
    PARTIALTORCH_API C10_ALWAYS_INLINE c10::intrusive_ptr<MaskedPair<T>> masked_pair(
            std::pair<T, std::conditional_t<
                    std::is_same_v<T, at::Tensor>,
                    c10::optional<at::Tensor>,
                    c10::optional<bool>>> args) {
        return at::make_intrusive<MaskedPair<T>>(MaskedPair<T>(args.first, args.second));
    }

    PARTIALTORCH_API C10_ALWAYS_INLINE c10::intrusive_ptr<MaskedPair<at::Tensor>> masked_pair(
            const at::ArrayRef<at::Tensor> args) {
        auto n_args = args.size();
        TORCH_CHECK_VALUE(1 <= n_args && n_args <= 2,
                          "MaskedPair must be initialized with a single tensor (data) "
                          "or a pair of tensors (data and mask). Got ",
                          n_args,
                          " arguments.")
        auto mask = (n_args == 2) ?  args[1] : c10::optional<at::Tensor>{};
        return at::make_intrusive<MaskedPair<at::Tensor>>(MaskedPair<at::Tensor>(args[0], mask));
    }

    using TensorMaskedPair = MaskedPair<at::Tensor>;
    using TensorMaskedPairList = at::ArrayRef<TensorMaskedPair>;
    using TensorMaskedPairIntrusivePtrList = at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>;
}

namespace std {
    template<size_t _Idx, typename T>
    struct masked_pair_element {
    };

    template<typename T>
    struct masked_pair_element<0, T> {
        using type = T;
    };

    template<typename T>
    struct masked_pair_element<1, T> {
        using type = std::conditional_t<
                std::is_same_v<T, at::Tensor>,
                c10::optional<at::Tensor>,
                c10::optional<bool>>;
    };

    template<size_t _Idx, class _Tuple>
    using masked_pair_element_t = typename masked_pair_element<_Idx, _Tuple>::type;

    template<class _Ret, typename T>
    constexpr _Ret _MaskedPair_get(
            const partialtorch::MaskedPair<T> &_Pr,
            integral_constant<size_t, 0>) noexcept {
        return _Pr.data_;
    }

    template<class _Ret, typename T>
    constexpr _Ret _MaskedPair_get(
            const c10::intrusive_ptr<partialtorch::MaskedPair<T>> &_Pr,
            integral_constant<size_t, 0>) noexcept {
        return _Pr->data_;
    }

    template<class _Ret, typename T>
    constexpr _Ret _MaskedPair_get(
            const partialtorch::MaskedPair<T> &_Pr,
            integral_constant<size_t, 1>) noexcept {
        return _Pr.mask_;
    }

    template<class _Ret, typename T>
    constexpr _Ret _MaskedPair_get(
            const c10::intrusive_ptr<partialtorch::MaskedPair<T>> &_Pr,
            integral_constant<size_t, 1>) noexcept {
        return _Pr->mask_;
    }

    template<size_t _Idx, typename T>
    constexpr masked_pair_element_t<_Idx, T> &get(
            const partialtorch::MaskedPair<T> &_Pr) noexcept {
        using _Rtype = masked_pair_element_t<_Idx, T> &;
        return _MaskedPair_get<_Rtype>(_Pr, integral_constant<size_t, _Idx>{});
    }

    template<size_t _Idx, typename T>
    constexpr masked_pair_element_t<_Idx, T> &get(
            const c10::intrusive_ptr<partialtorch::MaskedPair<T>> &_Pr) noexcept {
        using _Rtype = masked_pair_element_t<_Idx, T> &;
        return _MaskedPair_get<_Rtype>(_Pr, integral_constant<size_t, _Idx>{});
    }
}
