#pragma once

#include <string>

#include <ATen/ATen.h>

#include "../../MaskedPair.h"
#include "type_utils.h"

// ~~~~ macros for code gen (unused) ~~~~
#define PT_STRINGIFY(s) #s
#define PT_ESTRINGIFY(s) PT_STRINGIFY(s)

#define PT_GLUE(_0, _1) _0##_1
#define PT_EGLUE(_0, _1) PT_GLUE(_0, _1)

#define PT_WIDEN(_0) PT_EGLUE(L, _0)  // Widen string literal

#define PT_GET(_0, _1) _0  // Return the first of two arguments
#define PT_GET_(_0, _1) _1  // Return the second of two arguments

#define PT_JOIN(_0, _1) _0 ## _1  // Concatenate two arguments
#define PT_EJOIN(_0, _1) PT_JOIN(_0, _1)  // Expand macros and concatenate

#define PT_FIRST(_, ...) _  // Truncate everything after first comma
#define PT_EFIRST(_) PT_FIRST(_)  // Expand argument and pass to PT_FIRST

#define PT_REST(_0, ...) __VA_ARGS__  // Remove everything before first comma

#define PT_GET_GET(...) \
    PT_EJOIN(PT_GET, PT_EFIRST(PT_REST(,,##__VA_ARGS__ _)))  // Branch between GET and GET_

#define PT_IFARGS(YES, NO, ...) PT_GET_GET(__VA_ARGS__)(YES, NO)

#define PT_PREPEND_NOTHING()

#define PT_PREPEND_COMMA(...) , __VA_ARGS__
#define PT_PREPEND_COMMA_IF_NONEMPTY(...) PT_IFARGS(PT_PREPEND_COMMA, PT_PREPEND_NOTHING, __VA_ARGS__)(__VA_ARGS__)

#define PT_PREPEND_DOT(...) .##__VA_ARGS__
#define PT_PREPEND_DOT_IF_NONEMPTY(...) PT_IFARGS(PT_PREPEND_DOT, PT_PREPEND_NOTHING, __VA_ARGS__)(__VA_ARGS__)

#define PT_PREPEND_EQ(...) = __VA_ARGS__
#define PT_PREPEND_EQ_IF_NONEMPTY(...) PT_IFARGS(PT_PREPEND_EQ, PT_PREPEND_NOTHING, __VA_ARGS__)(__VA_ARGS__)

#define PT_PREPEND_DASH(...) _##__VA_ARGS__
#define PT_PREPEND_DASH_IF_NONEMPTY(...) PT_IFARGS(PT_PREPEND_DASH, PT_PREPEND_NOTHING, __VA_ARGS__)(__VA_ARGS__)

#define PT_PREPEND_EMPTY_STR(...) #__VA_ARGS__

#define PT_PREPEND_COMMA_STR(...) ","#__VA_ARGS__
#define PT_PREPEND_COMMA_STR_IF_NONEMPTY(...) PT_IFARGS(PT_PREPEND_COMMA_STR, PT_PREPEND_EMPTY_STR, __VA_ARGS__)(__VA_ARGS__)

#define PT_PREPEND_DOT_STR(...) "."#__VA_ARGS__
#define PT_PREPEND_DOT_STR_IF_NONEMPTY(...) PT_IFARGS(PT_PREPEND_DOT_STR, PT_PREPEND_EMPTY_STR, __VA_ARGS__)(__VA_ARGS__)

#define PT_PREPEND_DASH_STR(...) "_"#__VA_ARGS__
#define PT_PREPEND_DASH_STR_IF_NONEMPTY(...) PT_IFARGS(PT_PREPEND_DASH_STR, PT_PREPEND_EMPTY_STR, __VA_ARGS__)(__VA_ARGS__)

#define PT_PREPEND_EQ_STR(...) "="#__VA_ARGS__
#define PT_PREPEND_EQ_STR_IF_NONEMPTY(...) PT_IFARGS(PT_PREPEND_EQ_STR, PT_PREPEND_EMPTY_STR, __VA_ARGS__)(__VA_ARGS__)

namespace partialtorch {
    namespace ops {
        namespace utils {
            template<typename T>
            C10_ALWAYS_INLINE std::string type_schema_str() {
                // TODO: We have not covered a lot of possible scenarios.
                //  Also, should this function be constexpr?
#define RETURN_WITH_OPTIONAL(result) if constexpr (c10::is_optional_v<T>) {return result "?";} else {return result;}
                using base_t = typename c10::base_t<T>;
                if constexpr (std::is_same_v<base_t, TensorMaskedPair>) {
                    RETURN_WITH_OPTIONAL(TENSORMASKEDPAIR_SCHEMA_STR)
                } else if constexpr (std::is_same_v<base_t, at::Tensor>) {
                    RETURN_WITH_OPTIONAL("Tensor")
                } else if constexpr (std::is_same_v<base_t, at::OptionalTensorRef>) {
                    return "Tensor?";
                } else if constexpr (std::is_same_v<base_t, at::Scalar>) {
                    RETURN_WITH_OPTIONAL("Scalar")
                } else if constexpr (std::is_same_v<base_t, int64_t>) {
                    RETURN_WITH_OPTIONAL("int")
                } else if constexpr (std::is_same_v<base_t, c10::SymInt>) {
                    RETURN_WITH_OPTIONAL("SymInt")
                } else if constexpr (std::is_same_v<base_t, double_t>) {
                    RETURN_WITH_OPTIONAL("float")
                } else if constexpr (std::is_same_v<base_t, bool>) {
                    RETURN_WITH_OPTIONAL("bool")
                } else if constexpr (std::is_same_v<base_t, std::string> ||
                                     std::is_same_v<base_t, c10::string_view>) {
                    RETURN_WITH_OPTIONAL("str")
                } else if constexpr (std::is_same_v<base_t, at::Dimname>) {
                    return "Dimname";
                } else if constexpr (std::is_same_v<base_t, at::TensorOptions>) {
                    RETURN_WITH_OPTIONAL("TensorOptions")
                } else if constexpr (std::is_same_v<base_t, at::Generator>) {
                    RETURN_WITH_OPTIONAL("Generator")
                } else if constexpr (std::is_same_v<base_t, at::ScalarType>) {
                    RETURN_WITH_OPTIONAL("ScalarType")
                } else if constexpr (std::is_same_v<base_t, at::Layout>) {
                    RETURN_WITH_OPTIONAL("Layout")
                } else if constexpr (std::is_same_v<base_t, at::Device>) {
                    RETURN_WITH_OPTIONAL("Device")
                } else if constexpr (std::is_same_v<base_t, at::MemoryFormat>) {
                    RETURN_WITH_OPTIONAL("MemoryFormat")
                } else if constexpr (std::is_same_v<base_t, at::Stream>) {
                    RETURN_WITH_OPTIONAL("Stream")
                } else if constexpr (std::is_same_v<base_t, at::Storage>) {
                    RETURN_WITH_OPTIONAL("Storage")
                } else if constexpr (std::is_same_v<base_t, c10::QScheme>) {
                    RETURN_WITH_OPTIONAL("QScheme")
                }
                    // Containers: std::vector<T>, c10::ArrayRef<T>, c10::OptionalArrayRef<T>, c10::IListRef<T>
                    // TODO: handle all special ArrayRef cases
                else if constexpr (std::is_same_v<base_t, c10::IntArrayRef>) {
                    return "int[1]";
                } else if constexpr (std::is_same_v<base_t, c10::OptionalIntArrayRef>) {
                    return "int[1]?";
                } else if constexpr (std::is_same_v<base_t, c10::SymIntList>) {
                    return "SymInt[1]";
                } else if constexpr (std::is_same_v<base_t, at::OptionalSymIntArrayRef>) {
                    return "SymInt[1]?";
                } else if constexpr (std::is_same_v<base_t, at::DimnameList>) {
                    RETURN_WITH_OPTIONAL("DimnameList")
                } else if constexpr (std::is_vector_v<base_t> ||
                                     c10::is_arrayref_v<base_t> ||
                                     c10::is_list_v<base_t> ||
                                     c10::is_ilistref_v<base_t>) {
                    RETURN_WITH_OPTIONAL(type_schema_str<typename base_t::value_type>() + "[]")
                } else if constexpr (c10::is_optional_arrayref_v<base_t>) {
                    static_assert(std::is_pod_v<T>, "value_type of c10::OptionalArrayRef cannot be deduced.");
                } else if constexpr (std::is_same_v<base_t, void>) {
                    return "()";
                } else if constexpr (std::is_same_v<base_t, nullptr_t>) {
                    return "NoneType";
                } else {
                    static_assert(std::is_pod_v<T>, "Type not handled.");
                }
#undef RETURN_WITH_OPTIONAL
            }

            struct ArgumentSchemaBuilder {
                std::string type_;
                std::string name_;
                std::string default_value_;
                bool is_vararg_;

                ArgumentSchemaBuilder(
                        const std::string_view &type,
                        const std::string_view &name,
                        const std::string_view &default_value = "",
                        bool is_vararg = false) :
                        type_(type), name_(name), default_value_(default_value), is_vararg_(is_vararg) {}

                C10_NODISCARD inline std::string str() const {
                    if (is_vararg_)
                        return c10::str("*", name_);
                    if (default_value_.empty())
                        return c10::str(type_, " ", name_);
                    return c10::str(type_, " ", name_, "=", default_value_);
                }

                friend inline std::ostream &operator<<(std::ostream &stream, const ArgumentSchemaBuilder &self) {
                    return stream << self.str();
                }
            };

            struct ReturnSchemaBuilder {
                std::string type_;
                std::string name_;

                ReturnSchemaBuilder(
                        const std::string_view &type,
                        const std::string_view &name = "") :
                        type_(type), name_(name) {}

                C10_NODISCARD inline std::string str() const {
                    if (!name_.empty())
                        return c10::str(type_, " ", name_);
                    return type_;
                }

                friend inline std::ostream &operator<<(std::ostream &stream, const ReturnSchemaBuilder &self) {
                    return stream << self.str();
                }
            };

            struct FunctionSchemaBuilder {
                STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(namespace_, "partialtorch")
                std::string name_;
                std::vector<std::string> overload_tokens_;
                std::vector<ArgumentSchemaBuilder> args_;
                int64_t varargs_index_ = -1;
                std::vector<ReturnSchemaBuilder> returns_;

                FunctionSchemaBuilder() {}

                FunctionSchemaBuilder(
                        const std::string_view &name,
                        const std::string_view &overload_name = "") : name_(name) {
                    overload(overload_name);
                }

                inline FunctionSchemaBuilder &named(const std::string_view &name) {
                    name_ = name;
                    return *this;
                }

                inline FunctionSchemaBuilder &add_overload(const std::string_view &overload_token) {
                    if (!overload_token.empty())
                        overload_tokens_.emplace_back(overload_token);
                    return *this;
                }

                inline FunctionSchemaBuilder &overload(const std::string_view &overload_name) {
                    overload_tokens_.clear();
                    return add_overload(overload_name);
                }

                inline FunctionSchemaBuilder &vararg(c10::optional<int64_t> index = {}) {
                    if (index.has_value()) {
                        if (index.value() >= 0)
                            varargs_index_ = index.value();
                    } else {
                        varargs_index_ = args_.size();
                    }
                    return *this;
                }

                inline FunctionSchemaBuilder &arg(const std::string_view type,
                                                  const std::string_view &name,
                                                  const std::string_view &default_value = "") {
                    args_.emplace_back(type, name, default_value);
                    return *this;
                }

                template<typename T>
                inline FunctionSchemaBuilder &arg(const std::string &name,
                                                  const std::string &default_value = "") {
                    return arg(type_schema_str<T>(), name, default_value);
                }

                inline FunctionSchemaBuilder &ret(const std::string_view type,
                                                  const std::string_view &name = "") {
                    returns_.emplace_back(type, name);
                    return *this;
                }

                template<typename T>
                inline FunctionSchemaBuilder &ret(const std::string &name = "") {
                    if constexpr (std::is_same_v<T, void>) {
                        returns_.clear();
                        return *this;
                    } else
                        return ret(type_schema_str<T>(), name);
                }

                C10_NODISCARD inline std::string name() const {
                    auto schema_str = c10::str(namespace_, "::", name_.empty() ? "<anonymous_op>" : name_);
                    if (!overload_tokens_.empty()) {
                        auto overload_name = std::accumulate(
                                std::begin(overload_tokens_), std::end(overload_tokens_), std::string(),
                                [](const std::string &ss, const std::string &s) {
                                    return ss.empty() ? s : (s.empty() ? ss : ss + "_" + s);
                                });
                        if (!overload_name.empty())
                            schema_str += c10::str(".", overload_name);
                    }
                    return schema_str;
                }

                C10_NODISCARD inline std::string signature() const {
                    return c10::str(args_str(), returns_str());
                }

                C10_NODISCARD inline std::string schema() const {
                    auto res = c10::str(name(), signature());
#ifdef DEBUG_OPS_SCHEMAS
                    std::cout << c10::str("- func: ", res) << std::endl;
#endif
                    return res;
                }

                C10_NODISCARD inline std::string str() const {
                    return schema();
                }

                friend inline std::ostream &operator<<(std::ostream &stream, const FunctionSchemaBuilder &self) {
                    return stream << self.str();
                }

            private:
                C10_NODISCARD inline std::string args_str() const {
                    std::string ss;
                    for (const auto i : c10::irange(args_.size())) {
                        if (i > 0)
                            ss += ", ";
                        if (i == varargs_index_)
                            ss += "*, ";
                        ss += args_[i].str();
                    }
                    return c10::str("(", ss, ")");
                }

                C10_NODISCARD inline std::string returns_str() const {
                    if (returns_.empty())
                        return c10::str(" -> ", type_schema_str<void>());
                    auto ret = std::accumulate(
                            std::begin(returns_), std::end(returns_), std::string(),
                            [](const std::string &ss, const ReturnSchemaBuilder &ret) {
                                return ss.empty() ? ret.str() : ss + ", " + ret.str();
                            });
                    if (returns_.size() > 1)
                        return c10::str(" -> (", ret, ")");
                    return c10::str(" -> ", ret);
                }
            };
        }
    }
}
