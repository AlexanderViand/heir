#ifndef LIB_RUNTIME_CHEDDARRUNTIME_H_
#define LIB_RUNTIME_CHEDDARRUNTIME_H_

#include <array>
#include <cstddef>
#include <memory>

namespace heir {

template <typename T>
struct CArrayType {
  using type = T;
  using element_type = T;
};

template <typename T, std::size_t N>
struct CArrayType<std::array<T, N>> {
  using type = typename CArrayType<T>::type[N];
  using element_type = typename CArrayType<T>::element_type;
};

template <typename T>
using CArrayTypeT = typename CArrayType<T>::type;

template <typename T>
using CArrayElementTypeT = typename CArrayType<T>::element_type;

template <typename Context>
decltype(auto) getEncoder(Context& context) {
  return (context.encoder_);
}

template <typename T>
T* getPointer(std::unique_ptr<T>& value) {
  return value.get();
}

template <typename T>
T* data(T& value) {
  return &value;
}

template <typename T, std::size_t N>
CArrayElementTypeT<T>* data(std::array<T, N>& value) {
  return reinterpret_cast<CArrayElementTypeT<T>*>(value.data());
}

}  // namespace heir

#endif  // LIB_RUNTIME_CHEDDARRUNTIME_H_
