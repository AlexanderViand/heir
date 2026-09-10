#ifndef INCLUDE_HEIR_RUNTIME_CHEDDARRUNTIME_H_
#define INCLUDE_HEIR_RUNTIME_CHEDDARRUNTIME_H_

namespace heir {

template <typename Context>
decltype(auto) getEncoder(Context& context) {
  return (context.encoder_);
}

template <typename Context>
decltype(auto) getEncoder(Context* context) {
  return (context->encoder_);
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

}  // namespace heir

#endif  // INCLUDE_HEIR_RUNTIME_CHEDDARRUNTIME_H_
