#pragma once

#include <iterator>
#include <type_traits>

namespace c10 {
    namespace detail {
        template<typename T, typename I,
                typename std::enable_if_t<std::is_integral<I>::value, bool> = true>
        struct repeat_iterator {
            using iterator_category = std::input_iterator_tag;
            using value_type = T;
            using difference_type = std::ptrdiff_t;
            using pointer = T*;
            using reference = T&;

            explicit repeat_iterator(const T &value, I idx) : value(value), idx(idx) {}

            T operator*() const {
                return value;
            }

            T const *operator->() const {
                return &value;
            }

            repeat_iterator &operator++() {
                ++idx;
                return *this;
            }

            repeat_iterator operator++(int) {
                const auto copy = *this;
                ++*this;
                return copy;
            }

            bool operator==(const repeat_iterator &other) const {
                return value == other.value && (other.idx < 0 || idx == other.idx);
            }

            bool operator!=(const repeat_iterator &other) const {
                return value != other.value || !(*this == other);
            }

        protected:
            const T value;
            I idx;
        };

    } // namespace detail

    template<typename T, typename I,
            typename std::enable_if<std::is_integral<I>::value, bool>::type = true>
    struct irepeat {
    public:
        irepeat(const T &value, I n = static_cast<I>(1)) : begin_(value, 0), end_(value, n) {}

        using iterator = detail::repeat_iterator<T, I>;

        iterator begin() const {
            return begin_;
        }

        iterator end() const {
            return end_;
        }

    private:
        iterator begin_;
        iterator end_;
    };
}
