#pragma once

#include <Python.h>
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "MaskedPair.h"

namespace partialtorch {
    // pybind11 trampoline class of partialtorch::MaskedPair
    template<typename T>
    class [[maybe_unused]] PyMaskedPair : public MaskedPair<T> {
    public:
        using MaskedPair<T>::MaskedPair;
    };

    using PyTensorMaskedPair = PyMaskedPair<at::Tensor>;

    void initPyMaskedPair(PyObject *module) {
        auto py_module = py::reinterpret_borrow<py::module>(module);

        auto masked_pair_cls = py::class_<
                TensorMaskedPair,
                c10::intrusive_ptr<TensorMaskedPair>,
                PyTensorMaskedPair>(py_module, "MaskedPair")
                .def(py::init<at::Tensor &, const c10::optional<at::Tensor> &>(),
                     py::arg("data"), py::arg("mask") = py::none())
                .def_property("data", &TensorMaskedPair::get_data, &TensorMaskedPair::set_data)
                .def_property("mask", &TensorMaskedPair::get_mask, &TensorMaskedPair::set_mask)
                .def_property("members", &TensorMaskedPair::get_members, &TensorMaskedPair::set_members)
                .def("__getstate__",
                     [](const c10::intrusive_ptr<TensorMaskedPair> &self)
                             -> std::tuple<at::Tensor, c10::optional<at::Tensor>> {
                         return std::make_tuple(self->data_, self->mask_);
                     })
                .def("__setstate__",
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
                .def_property_readonly("shape", &TensorMaskedPair::sizes)
                .def_property_readonly("names", &TensorMaskedPair::names)
                .def("ndimension", &TensorMaskedPair::ndimension)
                .def_property_readonly("ndim", &TensorMaskedPair::ndimension)
                .def("is_contiguous", [](const c10::intrusive_ptr<TensorMaskedPair> &self) {
                    return self->is_contiguous();
                })
                .def("numel", &TensorMaskedPair::numel)
                .def("element_size", &TensorMaskedPair::element_size)
                .def_property_readonly("dtype", &TensorMaskedPair::scalar_type)
                .def("storage", &TensorMaskedPair::storage)
                .def("_is_zerotensor", &TensorMaskedPair::_is_zerotensor)
                .def("is_conj", &TensorMaskedPair::is_conj)
                .def("is_neg", &TensorMaskedPair::is_neg)
                .def_property_readonly("layout", &TensorMaskedPair::layout)
                .def_property_readonly("device", &TensorMaskedPair::device)
                .def("get_device", &TensorMaskedPair::get_device)
                .def_property_readonly("is_cpu", &TensorMaskedPair::is_cpu)
                .def_property_readonly("is_cuda", &TensorMaskedPair::is_cuda)
                .def_property_readonly("is_ipu", &TensorMaskedPair::is_ipu)
                .def_property_readonly("is_xpu", &TensorMaskedPair::is_xpu)
                .def_property_readonly("is_sparse", &TensorMaskedPair::is_sparse)
                .def_property_readonly("is_sparse_csr", &TensorMaskedPair::is_sparse_csr)
                .def_property_readonly("is_mkldnn", &TensorMaskedPair::is_mkldnn)
                .def_property_readonly("is_mps", &TensorMaskedPair::is_mps)
                .def_property_readonly("is_ort", &TensorMaskedPair::is_ort)
                .def_property_readonly("is_vulkan", &TensorMaskedPair::is_vulkan)
                .def_property_readonly("is_quantized", &TensorMaskedPair::is_quantized)
                .def_property_readonly("is_meta", &TensorMaskedPair::is_meta)
                .def("is_inference", &TensorMaskedPair::is_inference)
                .def_property_readonly("is_nested", &TensorMaskedPair::is_nested)
                .def("has_names", &TensorMaskedPair::has_names)
                .def_property_readonly("requires_grad", &TensorMaskedPair::requires_grad)
//                .def_property_readonly("grad_fn", &TensorMaskedPair::grad_fn)
                .def_property_readonly("is_leaf", &TensorMaskedPair::is_leaf)
                .def_property_readonly("output_nr", &TensorMaskedPair::output_nr)
                .def_property_readonly("_version", &TensorMaskedPair::_version)
                .def("retain_grad", &TensorMaskedPair::retain_grad)
                .def_property_readonly("retains_grad", &TensorMaskedPair::retains_grad)
                .def("requires_grad_", &TensorMaskedPair::requires_grad_, py::arg("_requires_grad") = true)
                .def_property_readonly("name", &TensorMaskedPair::name)
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
                .def("cpu", [](const c10::intrusive_ptr<TensorMaskedPair> &self) {
                    return self->cpu();
                })
                .def("cuda", [](const c10::intrusive_ptr<TensorMaskedPair> &self) {
                    return self->cuda();
                })
                .def("backward", &TensorMaskedPair::backward,
                     py::arg("gradient") = at::Tensor(),
                     py::arg("retain_graph") = py::none(),
                     py::arg("create_graph") = false,
                     py::arg("inputs") = py::none())
                .def("clone", [](const c10::intrusive_ptr<TensorMaskedPair> &self) {
                    return self->clone();
                })
                .def("contiguous", [](const c10::intrusive_ptr<TensorMaskedPair> &self) {
                    return self->contiguous();
                })
                .def("detach", &TensorMaskedPair::detach)
                .def("detach_", &TensorMaskedPair::detach_)
                .def("fill_masked", static_cast<c10::intrusive_ptr<TensorMaskedPair> (TensorMaskedPair::*)(
                        const at::Scalar &) const>(&TensorMaskedPair::fill_masked))
                .def("fill_masked_", static_cast<c10::intrusive_ptr<TensorMaskedPair> (TensorMaskedPair::*)(
                        const at::Scalar &)>(&TensorMaskedPair::fill_masked_))
                .def("to", [](const c10::intrusive_ptr<TensorMaskedPair> &self,
                              c10::optional<at::ScalarType> dtype,
                              c10::optional<at::Layout> layout,
                              c10::optional<at::Device> device,
                              c10::optional<bool> pin_memory,
                              bool non_blocking,
                              bool copy) {
                         return self->to(dtype, layout, device, pin_memory, non_blocking, copy, {});
                     },
                     py::kw_only(),
                     py::arg("dtype") = py::none(),
                     py::arg("layout") = py::none(),
                     py::arg("device") = py::none(),
                     py::arg("pin_memory") = py::none(),
                     py::arg("non_blocking") = false,
                     py::arg("copy") = false)
                .def("to_tensor", &TensorMaskedPair::to_tensor)
                .def("item", &TensorMaskedPair::item)
                .def("index_non_masked", &TensorMaskedPair::index_non_masked)
                .def("view", &TensorMaskedPair::view)
                .def("t", &TensorMaskedPair::t)
                .def("t_", &TensorMaskedPair::t_)
                .def("transpose", &TensorMaskedPair::transpose)
                .def("transpose_", &TensorMaskedPair::transpose_)
                .def("permute", &TensorMaskedPair::permute)
                .def("__getitem__", &TensorMaskedPair::__getitem__)
                .def("__next__", &TensorMaskedPair::__next__)
                .def("__iter__", &TensorMaskedPair::__iter__)
                .def("__str__", &TensorMaskedPair::__str__)
                .def("__repr__", &TensorMaskedPair::__repr__);
    }
}
