#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "math.h"
#include <cuda_runtime.h>
#include <limits>
#include <vector>

template <typename T> __global__ void sinKernel(double *result, T input) {
  result[0] = static_cast<double>(xtd::sin(input));
}

template <typename T> __global__ void sinfKernel(double *result, T input) {
  result[0] = static_cast<double>(xtd::sinf(input));
}

TEST_CASE("sinCuda", "[sin]") {
  int deviceCount;
  cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

  if (cudaStatus != cudaSuccess || deviceCount == 0) {
    exit(EXIT_SUCCESS);
  }

  cudaSetDevice(0);
  cudaStream_t q;
  cudaStreamCreate(&q);

  // input
  std::vector<double> values{-1., 0., M_PI / 2, M_PI, 42.};

  double* result;
  int constexpr N = 6;
  cudaMallocAsync(&result, N * sizeof(double), q);

  for (auto v : values) {

    cudaMemsetAsync(&result, 0x00, N * sizeof(double), q);

    sinKernel<<<1, 1, 0, q>>>(&result[0], static_cast<int>(v));
    sinKernel<<<1, 1, 0, q>>>(&result[1], static_cast<float>(v));
    sinKernel<<<1, 1, 0, q>>>(&result[2], static_cast<double>(v));
    sinfKernel<<<1, 1, 0, q>>>(&result[3], static_cast<int>(v));
    sinfKernel<<<1, 1, 0, q>>>(&result[4], static_cast<float>(v));
    sinfKernel<<<1, 1, 0, q>>>(&result[5], static_cast<double>(v));

    double resultHost[N];
    cudaMemcpyAsync(resultHost, result, N * sizeof(double), cudaMemcpyDeviceToHost, q);

    cudaStreamSynchronize(q);

    auto const epsilon = std::numeric_limits<double>::epsilon();
    auto const epsilon_f = std::numeric_limits<float>::epsilon();
    REQUIRE_THAT(resultHost[0], Catch::Matchers::WithinAbs(std::sin(static_cast<int>(v)), epsilon));
    REQUIRE_THAT(resultHost[1], Catch::Matchers::WithinAbs(std::sin(v), epsilon_f));
    REQUIRE_THAT(resultHost[2], Catch::Matchers::WithinAbs(std::sin(v), epsilon));
    REQUIRE_THAT(resultHost[3], Catch::Matchers::WithinAbs(sinf(static_cast<int>(v)), epsilon_f));
    REQUIRE_THAT(resultHost[4], Catch::Matchers::WithinAbs(sinf(v), epsilon_f));
    REQUIRE_THAT(resultHost[5], Catch::Matchers::WithinAbs(sinf(v), epsilon_f));
  }

  cudaFreeAsync(result, q);
}
