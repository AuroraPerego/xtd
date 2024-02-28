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

  double* result;
  int constexpr N = 4;
  hipMallocAsync(&result, N * sizeof(double), q);
{
  std::ifstream inputFile("Regr/sin/solutions.txt");
  double inp, sol;
  while (inputFile >> inp >> sol) {

    hipMemsetAsync(&result, 0x00, N * sizeof(double), q);

    sinKernel<<<1, 1, 0, q>>>(&result[0], static_cast<float>(inp));
    sinKernel<<<1, 1, 0, q>>>(&result[1], static_cast<double>(inp));
    sinfKernel<<<1, 1, 0, q>>>(&result[2], static_cast<float>(inp));
    sinfKernel<<<1, 1, 0, q>>>(&result[3], static_cast<double>(inp));

    double resultHost[N];
    hipMemcpyAsync(resultHost, result, N * sizeof(double), hipMemcpyDeviceToHost, q);

    hipStreamSynchronize(q);

    auto const epsilon = std::numeric_limits<double>::epsilon();
    auto const epsilon_f = std::numeric_limits<float>::epsilon();
    REQUIRE_THAT(resultHost[0], Catch::Matchers::WithinAbs(sol, epsilon_f));
    REQUIRE_THAT(resultHost[1], Catch::Matchers::WithinAbs(sol, epsilon));
    REQUIRE_THAT(resultHost[2], Catch::Matchers::WithinAbs(sol, epsilon_f));
    REQUIRE_THAT(resultHost[3], Catch::Matchers::WithinAbs(sol, epsilon_f));
  }
  inputFile.close();
}
{
  std::ifstream inputFile("Regr/sin/solutionsInt.txt");
  double inp, sol;
  while (inputFile >> inp >> sol) {

    hipMemsetAsync(&result, 0x00, N * sizeof(double), q);

    sinKernel<<<1, 1, 0, q>>>(&result[0], static_cast<int>(inp));
    sinfKernel<<<1, 1, 0, q>>>(&result[1], static_cast<int>(inp));

    double resultHost[N];
    hipMemcpyAsync(resultHost, result, N * sizeof(double), hipMemcpyDeviceToHost, q);

    hipStreamSynchronize(q);

    auto const epsilon = std::numeric_limits<double>::epsilon();
    auto const epsilon_f = std::numeric_limits<float>::epsilon();
    REQUIRE_THAT(resultHost[0], Catch::Matchers::WithinAbs(sol, epsilon));
    REQUIRE_THAT(resultHost[1], Catch::Matchers::WithinAbs(sol, epsilon_f));
  }
  inputFile.close();
}
  hipFreeAsync(result, q);
}
