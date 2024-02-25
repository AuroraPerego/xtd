#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "math.h"
#include <cmath>
#include <hip_runtime.h>
#include <limits>

template <typename T> __global__ void sinKernel(double *result, T input) {
  result[0] = static_cast<double>(xtd::sin(input));
}

template <typename T> __global__ void sinfKernel(double *result, T input) {
  result[0] = static_cast<double>(xtd::sinf(input));
}

TEST_CASE("sinHip", "[sin]") {
  int deviceCount;
  hipError_t hipStatus = hipGetDeviceCount(&deviceCount);

  if (hipStatus != hipSuccess || deviceCount == 0) {
    exit(EXIT_SUCCESS);
  }

  hipSetDevice(0);
  hipStream_t q;
  hipStreamCreate(&q);

  // input
  std::vector<double> values{-1., 0., M_PI / 2, M_PI, 42.};

  double* result;
  int constexpr N = 6;
  hipMallocAsync(&result, N * sizeof(double), q);

  for (auto v : values) {

    hipMemsetAsync(&result, 0x00, N * sizeof(double), q);

    sinKernel<<<1, 1, 0, q>>>(&result[0], static_cast<int>(v));
    sinKernel<<<1, 1, 0, q>>>(&result[1], static_cast<float>(v));
    sinKernel<<<1, 1, 0, q>>>(&result[2], static_cast<double>(v));
    sinfKernel<<<1, 1, 0, q>>>(&result[3], static_cast<int>(v));
    sinfKernel<<<1, 1, 0, q>>>(&result[4], static_cast<float>(v));
    sinfKernel<<<1, 1, 0, q>>>(&result[5], static_cast<double>(v));

    double resultHost[N];
    hipMemcpyAsync(resultHost, result, N * sizeof(double), hipMemcpyDeviceToHost, q);

    hipStreamSynchronize(q);

    auto const epsilon = std::numeric_limits<double>::epsilon();
    auto const epsilon_f = std::numeric_limits<float>::epsilon();
    REQUIRE_THAT(resultHost[0], Catch::Matchers::WithinAbs(std::sin(static_cast<int>(v)), epsilon));
    REQUIRE_THAT(resultHost[1], Catch::Matchers::WithinAbs(std::sin(v), epsilon_f));
    REQUIRE_THAT(resultHost[2], Catch::Matchers::WithinAbs(std::sin(v), epsilon));
    REQUIRE_THAT(resultHost[3], Catch::Matchers::WithinAbs(sinf(static_cast<int>(v)), epsilon_f));
    REQUIRE_THAT(resultHost[4], Catch::Matchers::WithinAbs(sinf(v), epsilon_f));
    REQUIRE_THAT(resultHost[5], Catch::Matchers::WithinAbs(sinf(v), epsilon_f));
  }
}
