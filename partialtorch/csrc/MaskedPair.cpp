#include "MaskedPair.h"
#include "formatting.h"

#include "utils/Exception.h"
#include "utils/pybind11.h"

#include "ops/utils/schema_utils.h"
#include "ops/ops.h"

namespace partialtorch {
    // ops
    inline void TensorMaskedPair::backward(
            const at::Tensor &gradient,
            c10::optional<bool> retain_graph,
            bool create_graph,
            c10::optional<at::TensorList> inputs) const {
        data_.backward(gradient.defined() && mask_.has_value() ? at::where(
                mask_.value(), gradient, 0) : gradient, retain_graph, create_graph, inputs);
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::clone(
            c10::optional<at::MemoryFormat> memory_format) const {
        return ops::clone(get_intrusive_ptr(), memory_format);
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::contiguous(
            at::MemoryFormat memory_format) const {
        return ops::contiguous(get_intrusive_ptr(), memory_format);
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::detach() const {
        return ops::detach(get_intrusive_ptr());
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::detach_() {
        return ops::detach_(get_intrusive_ptr());
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::fill_masked(const at::Scalar &value) const {
        return ops::fill_masked(get_intrusive_ptr(), value);
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::fill_masked(const at::Tensor &value) const {
        return ops::fill_masked(get_intrusive_ptr(), value);
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::fill_masked_(const at::Scalar &value) {
        return ops::fill_masked_(get_intrusive_ptr(), value);
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::fill_masked_(const at::Tensor &value) {
        return ops::fill_masked_(get_intrusive_ptr(), value);
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::to(
            at::TensorOptions options,
            bool non_blocking,
            bool copy,
            c10::optional<at::MemoryFormat> memory_format) const {
        return ops::to(get_intrusive_ptr(),
                       c10::optTypeMetaToScalarType(options.dtype_opt()),
                       options.layout_opt(), options.device_opt(),
                       options.pinned_memory_opt(),
                       non_blocking,
                       copy,
                       c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::to(
            c10::optional<at::ScalarType> dtype,
            c10::optional<at::Layout> layout,
            c10::optional<at::Device> device,
            c10::optional<bool> pin_memory,
            bool non_blocking,
            bool copy,
            c10::optional<at::MemoryFormat> memory_format) const {
        return ops::to(get_intrusive_ptr(), dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::to(
            at::Device device,
            at::ScalarType dtype,
            bool non_blocking,
            bool copy,
            c10::optional<at::MemoryFormat> memory_format) const {
        return ops::to(get_intrusive_ptr(), device, dtype, non_blocking, copy, memory_format);
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::to(
            at::ScalarType dtype,
            bool non_blocking,
            bool copy,
            c10::optional<at::MemoryFormat> memory_format) const {
        return ops::to(get_intrusive_ptr(), dtype, non_blocking, copy, memory_format);
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::to(
            const c10::intrusive_ptr<TensorMaskedPair> &other,
            bool non_blocking,
            bool copy,
            c10::optional<at::MemoryFormat> memory_format) const {
        return ops::to(get_intrusive_ptr(), other, non_blocking, copy, memory_format);
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::to(
            const at::Tensor &other,
            bool non_blocking,
            bool copy,
            c10::optional<at::MemoryFormat> memory_format) const {
        return ops::to(get_intrusive_ptr(), other, non_blocking, copy, memory_format);
    }

    inline at::Tensor TensorMaskedPair::to_tensor(const at::Scalar &value) const {
        return ops::to_tensor(get_intrusive_ptr(), value);
    }

    inline c10::intrusive_ptr<MaskedPair<at::Tensor>> TensorMaskedPair::index(
            at::ArrayRef<at::indexing::TensorIndex> indices) const {
        auto output_data = data_.index(indices);
        auto output_mask = mask_.has_value() ? mask_.value().index(indices)
                                             : c10::optional<at::Tensor>{};
        return masked_pair(output_data, output_mask);
    }

    inline at::Tensor TensorMaskedPair::index_non_masked() const {
        return ops::index_non_masked(get_intrusive_ptr());
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::t() const {
        return ops::t(get_intrusive_ptr());
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::t_() {
        return ops::t_(get_intrusive_ptr());
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::transpose(int64_t dim0, int64_t dim1) const {
        return ops::transpose(get_intrusive_ptr(), dim0, dim1);
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::transpose_(int64_t dim0, int64_t dim1) {
        return ops::transpose(get_intrusive_ptr(), dim0, dim1);
    }

    // Python methods
    inline c10::optional<at::Tensor> TensorMaskedPair::__getitem__(int64_t item) const {
        if (item == 0 || item == -2)
            return data_;
        if (item == 1 || item == -1)
            return mask_;
        TORCH_CHECK_INDEX(false, "Index ", item, " is out of bound.")
    }

    inline c10::optional<at::Tensor> TensorMaskedPair::__next__() {
        _idx += 1;
        if (_idx == 1)
            return data_;
        if (_idx == 2)
            return mask_;
        TORCH_CHECK_WITH(StopIteration, false, "")
    }

    inline c10::intrusive_ptr<TensorMaskedPair> TensorMaskedPair::__iter__() {
        _idx = 0;
        return c10::make_intrusive<TensorMaskedPair>(*this);
    }

    inline std::string TensorMaskedPair::__str__() const {
        std::stringstream ss;
        print(ss, *this);
        return ss.str();
    }

    inline std::string TensorMaskedPair::__repr__() const {
        return __str__();
    }

/// Identity creation api for python.
    template<typename T>
    C10_ALWAYS_INLINE c10::intrusive_ptr<MaskedPair<T>> &&masked_pair(
            c10::intrusive_ptr<MaskedPair<T>> &&pair) {
        return std::forward<c10::intrusive_ptr<MaskedPair<T>>>(pair);
    }

    TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
        m.class_<TensorMaskedPair>("MaskedPair")
                .def(torch::init<at::Tensor &, const c10::optional<at::Tensor> &>())
                .def_property("data", &TensorMaskedPair::get_data, &TensorMaskedPair::set_data)
                .def_property("mask", &TensorMaskedPair::get_mask, &TensorMaskedPair::set_mask)
                .def_property("members", &TensorMaskedPair::get_members, &TensorMaskedPair::set_members)
                .def_pickle(
                        [](const c10::intrusive_ptr<TensorMaskedPair> &self)
                                -> std::tuple<at::Tensor, c10::optional<at::Tensor>> {
                            return std::make_tuple(self->data_, self->mask_);
                        },
                        [](std::tuple<at::Tensor, c10::optional<at::Tensor>> state)
                                -> c10::intrusive_ptr<TensorMaskedPair> {
                            return at::make_intrusive<TensorMaskedPair>(
                                    TensorMaskedPair(std::get<0>(state), std::get<1>(state)));
                        })
                .def("dim", &TensorMaskedPair::dim)
                .def("storage_offset", &TensorMaskedPair::storage_offset)
                .def("is_complex", &TensorMaskedPair::is_complex)
                .def("is_floating_point", &TensorMaskedPair::is_floating_point)
                .def("is_signed", &TensorMaskedPair::is_signed)
                .def("size", &TensorMaskedPair::size)
                .def("stride", &TensorMaskedPair::strides)
                .def("sizes", &TensorMaskedPair::sizes)
                .def_property("shape", &TensorMaskedPair::sizes)
                .def_property("names", &TensorMaskedPair::names)
                .def("ndimension", &TensorMaskedPair::ndimension)
                .def_property("ndim", &TensorMaskedPair::ndimension)
                .def("is_contiguous", &TensorMaskedPair::is_contiguous, "", {
                        torch::arg("memory_format") = at::MemoryFormat::Contiguous})
                .def("numel", &TensorMaskedPair::numel)
                .def("element_size", &TensorMaskedPair::element_size)
                .def_property("dtype", &TensorMaskedPair::scalar_type)
                .def("storage", &TensorMaskedPair::storage)
                .def("_is_zerotensor", &TensorMaskedPair::_is_zerotensor)
                .def("is_conj", &TensorMaskedPair::is_conj)
                .def("is_neg", &TensorMaskedPair::is_neg)
                .def_property("layout", &TensorMaskedPair::layout)
                .def_property("device", &TensorMaskedPair::device)
                .def("get_device", &TensorMaskedPair::get_device)
                .def_property("is_cpu", &TensorMaskedPair::is_cpu)
                .def_property("is_cuda", &TensorMaskedPair::is_cuda)
                .def_property("is_ipu", &TensorMaskedPair::is_ipu)
                .def_property("is_xpu", &TensorMaskedPair::is_xpu)
                .def_property("is_sparse", &TensorMaskedPair::is_sparse)
                .def_property("is_sparse_csr", &TensorMaskedPair::is_sparse_csr)
                .def_property("is_mkldnn", &TensorMaskedPair::is_mkldnn)
                .def_property("is_mps", &TensorMaskedPair::is_mps)
                .def_property("is_ort", &TensorMaskedPair::is_ort)
                .def_property("is_vulkan", &TensorMaskedPair::is_vulkan)
                .def_property("is_quantized", &TensorMaskedPair::is_quantized)
                .def_property("is_meta", &TensorMaskedPair::is_meta)
                .def("is_inference", &TensorMaskedPair::is_inference)
                .def_property("is_nested", &TensorMaskedPair::is_nested)
                .def("has_names", &TensorMaskedPair::has_names)
                .def_property("requires_grad", &TensorMaskedPair::requires_grad)
//                .def_property("grad_fn", &TensorMaskedPair::grad_fn)
                .def_property("is_leaf", &TensorMaskedPair::is_leaf)
                .def_property("output_nr", &TensorMaskedPair::output_nr)
                .def_property("_version", &TensorMaskedPair::_version)
                .def("retain_grad", &TensorMaskedPair::retain_grad)
                .def_property("retains_grad", &TensorMaskedPair::retains_grad)
                .def("requires_grad_", &TensorMaskedPair::requires_grad_, "", {
                        torch::arg("_requires_grad") = true})
                .def_property("name", &TensorMaskedPair::name)
                .def_property(
                        "grad",
                        [](const c10::intrusive_ptr<TensorMaskedPair> &self) -> at::Tensor & {
                            return self->mutable_grad();
                        },
                        [](const c10::intrusive_ptr<TensorMaskedPair> &self, const at::Tensor &grad) -> void {
                            TORCH_CHECK(grad.sizes() == self->sizes(),
                                        "assigned grad has data of a different size")
                            TORCH_CHECK(grad.scalar_type() == self->scalar_type() &&
                                        grad.device() == self->device() &&
                                        grad.layout() == self->layout(),
                                        "assigned grad has data of a different type")
                            self->mutable_grad() = grad;
                        })
                .def("cpu", &TensorMaskedPair::cpu, "", {
                        torch::arg("memory_format") = at::MemoryFormat::Preserve})
                .def("cuda", &TensorMaskedPair::cuda, "", {
                        torch::arg("device") = at::Device(at::DeviceType::CUDA),
                        torch::arg("non_blocking") = false,
                        torch::arg("memory_format") = at::MemoryFormat::Preserve})
                .def("backward", &TensorMaskedPair::backward, "", {
                        torch::arg("gradient") = at::Tensor(),
                        torch::arg("retain_graph") = torch::arg::none(),
                        torch::arg("create_graph") = false,
                        torch::arg("inputs") = torch::arg::none()})
                .def("clone", &TensorMaskedPair::clone, "", {
                        torch::arg("memory_format") = at::MemoryFormat::Preserve})
                .def("contiguous", &TensorMaskedPair::contiguous, "", {
                        torch::arg("memory_format") = at::MemoryFormat::Preserve})
                .def("detach", &TensorMaskedPair::detach)
                .def("detach_", &TensorMaskedPair::detach_)
                .def("fill_masked", static_cast<c10::intrusive_ptr<TensorMaskedPair> (TensorMaskedPair::*)(
                        const at::Scalar &) const>(&TensorMaskedPair::fill_masked))
                .def("fill_masked_", static_cast<c10::intrusive_ptr<TensorMaskedPair> (TensorMaskedPair::*)(
                        const at::Scalar &)>(&TensorMaskedPair::fill_masked_))
                .def("to", static_cast<c10::intrusive_ptr<TensorMaskedPair> (TensorMaskedPair::*)(
                        c10::optional<at::ScalarType>,
                        c10::optional<at::Layout>,
                        c10::optional<at::Device>,
                        c10::optional<bool>,
                        bool, bool,
                        c10::optional<at::MemoryFormat>) const>(&TensorMaskedPair::to), "", {
                             torch::arg("dtype") = torch::arg::none(),
                             torch::arg("layout") = torch::arg::none(),
                             torch::arg("device") = torch::arg::none(),
                             torch::arg("pin_memory") = torch::arg::none(),
                             torch::arg("non_blocking") = false,
                             torch::arg("copy") = false,
                             torch::arg("memory_format") = torch::arg::none()})
                .def("to_tensor", &TensorMaskedPair::to_tensor)
                .def("item", &TensorMaskedPair::item)
                .def("index_non_masked", &TensorMaskedPair::index_non_masked)
                .def("t", &TensorMaskedPair::t)
                .def("t_", &TensorMaskedPair::t_)
                .def("transpose", &TensorMaskedPair::transpose)
                .def("transpose_", &TensorMaskedPair::transpose_)
                .def("__getitem__", &TensorMaskedPair::__getitem__)
                .def("__next__", &TensorMaskedPair::__next__)
                .def("__iter__", &TensorMaskedPair::__iter__)
                .def("__str__", &TensorMaskedPair::__str__)
                .def("__repr__", &TensorMaskedPair::__repr__);

        // Creation ops
        m.def(ops::utils::FunctionSchemaBuilder("masked_pair")
                      .arg<const at::Tensor &>("data")
                      .arg<const c10::optional<at::Tensor> &>("mask", "None")
                      .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
              static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                      const at::Tensor &, const c10::optional<at::Tensor> &)>(&masked_pair));
        m.def(ops::utils::FunctionSchemaBuilder("masked_pair").overload("bool")
                      .arg<const at::Tensor &>("data")
                      .arg<const c10::optional<bool>>("mask")
                      .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
              static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                      const at::Tensor &, const c10::optional<bool>)>(&masked_pair));
        m.def(ops::utils::FunctionSchemaBuilder("masked_pair").overload("tuple")
                      .arg<const at::ArrayRef<at::Tensor>>("args")
                      .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
              static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                      const at::ArrayRef<at::Tensor>)>(&masked_pair));
        m.def(ops::utils::FunctionSchemaBuilder("masked_pair").overload("identity")
                      .arg<c10::intrusive_ptr<TensorMaskedPair> &&>("self")
                      .ret<c10::intrusive_ptr<TensorMaskedPair> &&>().schema().c_str(),
              static_cast<c10::intrusive_ptr<TensorMaskedPair> &&(*)(
                      c10::intrusive_ptr<TensorMaskedPair> &&)>(&masked_pair));
    }
}
