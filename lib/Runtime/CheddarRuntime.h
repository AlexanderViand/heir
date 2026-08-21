#ifndef LIB_RUNTIME_CHEDDARRUNTIME_H_
#define LIB_RUNTIME_CHEDDARRUNTIME_H_

#include <array>
#include <cstddef>
#include <memory>

namespace heir {

template <typename T>
struct CArrayType {
  using type = T;
};

template <typename T, std::size_t N>
struct CArrayType<std::array<T, N>> {
  using type = typename CArrayType<T>::type[N];
};

template <typename T>
using CArrayTypeT = typename CArrayType<T>::type;

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
CArrayTypeT<T>* data(std::array<T, N>& value) {
  return reinterpret_cast<CArrayTypeT<T>*>(value.data());
}

}  // namespace heir

#endif  // LIB_RUNTIME_CHEDDARRUNTIME_H_
