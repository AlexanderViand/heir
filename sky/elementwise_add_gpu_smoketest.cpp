#include <cuda.h>

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

constexpr int kNumElements = 16384;
using Scalar = std::int64_t;

#define CU_CHECK(expr)                                                 \
  do {                                                                 \
    CUresult status = (expr);                                          \
    if (status != CUDA_SUCCESS) {                                      \
      const char* name = nullptr;                                      \
      const char* message = nullptr;                                   \
      cuGetErrorName(status, &name);                                   \
      cuGetErrorString(status, &message);                              \
      std::cerr << #expr << " failed: " << (name ? name : "<unknown>") \
                << " - " << (message ? message : "<unknown>") << '\n'; \
      return 1;                                                        \
    }                                                                  \
  } while (false)

int run(const char* cubin_path) {
  CU_CHECK(cuInit(0));

  CUdevice device;
  CU_CHECK(cuDeviceGet(&device, 0));

  CUcontext context;
  CU_CHECK(cuCtxCreate(&context, 0, device));

  CUmodule module;
  CU_CHECK(cuModuleLoad(&module, cubin_path));

  CUfunction function;
  CU_CHECK(cuModuleGetFunction(&function, module, "elementwise_add_kernel"));

  std::vector<Scalar> lhs(kNumElements);
  std::vector<Scalar> rhs(kNumElements);
  std::vector<Scalar> out(kNumElements, -1);

  for (int i = 0; i < kNumElements; ++i) {
    lhs[i] = (static_cast<Scalar>(i % 257) - 128) * 1099511627LL;
    rhs[i] = (static_cast<Scalar>((kNumElements - i) % 257) - 128) * 1048573LL;
  }

  CUdeviceptr d_lhs;
  CUdeviceptr d_rhs;
  CUdeviceptr d_out;
  CU_CHECK(cuMemAlloc(&d_lhs, sizeof(Scalar) * kNumElements));
  CU_CHECK(cuMemAlloc(&d_rhs, sizeof(Scalar) * kNumElements));
  CU_CHECK(cuMemAlloc(&d_out, sizeof(Scalar) * kNumElements));

  CU_CHECK(cuMemcpyHtoD(d_lhs, lhs.data(), sizeof(Scalar) * kNumElements));
  CU_CHECK(cuMemcpyHtoD(d_rhs, rhs.data(), sizeof(Scalar) * kNumElements));
  CU_CHECK(cuMemcpyHtoD(d_out, out.data(), sizeof(Scalar) * kNumElements));

  std::int64_t block_size = 1;
  std::int64_t offset = 0;
  std::int64_t size = kNumElements;
  std::int64_t stride = 1;
  void* params[] = {
      &block_size, &offset, &d_lhs,  &d_lhs,  &offset, &size,
      &stride,     &d_rhs,  &d_rhs,  &offset, &size,   &stride,
      &d_out,      &d_out,  &offset, &size,   &stride,
  };

  CU_CHECK(cuLaunchKernel(function, kNumElements, 1, 1, 1, 1, 1, 0, nullptr,
                          params, nullptr));
  CU_CHECK(cuCtxSynchronize());
  CU_CHECK(cuMemcpyDtoH(out.data(), d_out, sizeof(Scalar) * kNumElements));

  for (int i = 0; i < kNumElements; ++i) {
    const Scalar expected = lhs[i] + rhs[i];
    if (out[i] != expected) {
      std::cerr << "mismatch at " << i << ": got " << out[i] << ", expected "
                << expected << '\n';
      return 1;
    }
  }

  std::cout << "GPU smoke test passed\n";
  std::cout << "sample outputs: " << out[0] << ", " << out[1] << ", " << out[2]
            << '\n';

  CU_CHECK(cuMemFree(d_lhs));
  CU_CHECK(cuMemFree(d_rhs));
  CU_CHECK(cuMemFree(d_out));
  CU_CHECK(cuModuleUnload(module));
  CU_CHECK(cuCtxDestroy(context));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " KERNEL.cubin\n";
    return 1;
  }
  return run(argv[1]);
}
