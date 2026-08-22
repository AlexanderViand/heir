#ifndef INCLUDE_HEIR_RUNTIME_CHEDDARRUNTIME_H_
#define INCLUDE_HEIR_RUNTIME_CHEDDARRUNTIME_H_

#include <array>
#include <cstddef>
#include <memory>
#include <utility>

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

template <typename Context>
decltype(auto) getEncoder(Context* context) {
  return (context->encoder_);
}

template <typename T>
T* getPointer(std::unique_ptr<T>& value) {
  return value.get();
}

// Moves the fields into a freshly constructed aggregate.
template <typename Aggregate, typename... Field>
Aggregate pack(Field&... fields) {
  return Aggregate(std::move(fields)...);
}

template <typename T>
T* data(T& value) {
  return &value;
}

template <typename T>
const T* data(const T& value) {
  return &value;
}

// A key-only public key is a pointer to the evaluation-key map; the compute
// functions take the map by reference.
template <typename T>
const T& deref(const T* value) {
  return *value;
}

template <typename Keys, typename Context>
decltype(auto) multiplicationKey(const Keys& keys, Context& context) {
  if constexpr (requires { context.BootSecretId(); }) {
    return keys.GetMultiplicationKey(context.BootSecretId());
  } else {
    return keys.GetMultiplicationKey();
  }
}

template <typename Keys, typename Context>
decltype(auto) multiplicationKey(const Keys& keys, Context* context) {
  return multiplicationKey(keys, *context);
}

template <typename T, std::size_t N>
CArrayElementTypeT<T>* data(std::array<T, N>& value) {
  return reinterpret_cast<CArrayElementTypeT<T>*>(value.data());
}

template <typename T, std::size_t N>
const CArrayElementTypeT<T>* data(const std::array<T, N>& value) {
  return reinterpret_cast<const CArrayElementTypeT<T>*>(value.data());
}

}  // namespace heir

#endif  // INCLUDE_HEIR_RUNTIME_CHEDDARRUNTIME_H_
