#pragma once

#include <vector>
#include <type_traits>

#include <ATen/ATen.h>

namespace std {
    template<typename T>
    using base_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

    template<typename T>
    struct is_vector {
        static constexpr bool value = false;
    };

    template<typename T>
    struct is_vector<std::vector<T>> {
        static constexpr bool value = true;
    };

    template<typename T>
    constexpr bool is_vector_v = is_vector<T>::value;
}

namespace c10 {
    // c10::optional<T>
    template<typename T>
    struct is_optional {
        static constexpr bool value = false;
    };

    template<typename T>
    struct is_optional<c10::optional<T>> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<const c10::optional<T>> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<c10::optional<T> &> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<const c10::optional<T> &> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<c10::optional<T> &&> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<const c10::optional<T> &&> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<c10::optional<T> *> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<const c10::optional<T> *> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<c10::OptionalArrayRef<T>> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<const c10::OptionalArrayRef<T>> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<c10::OptionalArrayRef<T> &> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<const c10::OptionalArrayRef<T> &> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<c10::OptionalArrayRef<T> &&> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<const c10::OptionalArrayRef<T> &&> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<c10::OptionalArrayRef<T> *> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_optional<const c10::OptionalArrayRef<T> *> {
        static constexpr bool value = true;
    };

    template<typename T>
    constexpr bool is_optional_v = is_optional<T>::value;

    template<typename T>
    struct remove_optional {
        using type = T;
    };

    template<typename T>
    struct remove_optional<c10::optional<T>> {
        using type = typename c10::optional<T>::value_type;
    };

    template<typename T>
    struct remove_optional<c10::OptionalArrayRef<T>> {
        using type = typename c10::optional<c10::ArrayRef<T>>::value_type;
    };

    template<typename T>
    using remove_optional_t = typename remove_optional<T>::type;

    // c10::intrusive_ptr<T>
    template<typename T>
    struct is_intrusive {
        static constexpr bool value = false;
    };

    template<typename T>
    struct is_intrusive<c10::intrusive_ptr<T>> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_intrusive<const c10::intrusive_ptr<T>> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_intrusive<c10::intrusive_ptr<T> &> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_intrusive<const c10::intrusive_ptr<T> &> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_intrusive<c10::intrusive_ptr<T> &&> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_intrusive<const c10::intrusive_ptr<T> &&> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_intrusive<c10::intrusive_ptr<T> *> {
        static constexpr bool value = true;
    };

    template<typename T>
    struct is_intrusive<const c10::intrusive_ptr<T> *> {
        static constexpr bool value = true;
    };

    template<typename T>
    constexpr bool is_intrusive_v = is_intrusive<T>::value;

    template<typename T>
    struct remove_intrusive {
        using type = T;
    };

    template<typename T>
    struct remove_intrusive<c10::intrusive_ptr<T>> {
        using type = typename c10::intrusive_ptr<T>::element_type;
    };

    template<typename T>
    using remove_intrusive_t = typename remove_intrusive<T>::type;

    template<typename T>
    using base_t = typename remove_optional<typename remove_intrusive<typename std::base_t<T>>::type>::type;

    // c10::ArrayRef<T>
    template<typename T>
    struct is_arrayref {
        static constexpr bool value = false;
    };

    template<typename T>
    struct is_arrayref<c10::ArrayRef<T>> {
        static constexpr bool value = true;
    };

    template<typename T>
    constexpr bool is_arrayref_v = is_arrayref<T>::value;

    // c10::OptionalArrayRef<T>
    template<typename T>
    struct is_optional_arrayref {
        static constexpr bool value = false;
    };

    template<typename T>
    struct is_optional_arrayref<c10::OptionalArrayRef<T>> {
        static constexpr bool value = true;
    };

    template<typename T>
    constexpr bool is_optional_arrayref_v = is_optional_arrayref<T>::value;

    // c10::IListRef<T>
    template<typename T>
    struct is_list {
        static constexpr bool value = false;
    };

    template<typename T>
    struct is_list<c10::List<T>> {
        static constexpr bool value = true;
    };

    template<typename T>
    constexpr bool is_list_v = is_list<T>::value;

    // c10::IListRef<T>
    template<typename T>
    struct is_ilistref {
        static constexpr bool value = false;
    };

    template<typename T>
    struct is_ilistref<c10::IListRef<T>> {
        static constexpr bool value = true;
    };

    template<typename T>
    constexpr bool is_ilistref_v = is_ilistref<T>::value;
}
