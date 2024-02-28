#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "math.h"
#include <cuda_runtime.h>
#include <fstream>
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

  double* result;
  int constexpr N = 4;
  cudaMallocAsync(&result, N * sizeof(double), q);
{
  std::ifstream inputFile("Regr/sin/solutions.txt");
  double inp, sol;
  while (inputFile >> inp >> sol) {

    cudaMemsetAsync(&result, 0x00, N * sizeof(double), q);

    sinKernel<<<1, 1, 0, q>>>(&result[0], static_cast<float>(inp));
    sinKernel<<<1, 1, 0, q>>>(&result[1], static_cast<double>(inp));
    sinfKernel<<<1, 1, 0, q>>>(&result[2], static_cast<float>(inp));
    sinfKernel<<<1, 1, 0, q>>>(&result[3], static_cast<double>(inp));

    double resultHost[N];
    cudaMemcpyAsync(resultHost, result, N * sizeof(double), cudaMemcpyDeviceToHost, q);

    cudaStreamSynchronize(q);

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

    cudaMemsetAsync(&result, 0x00, N * sizeof(double), q);

    sinKernel<<<1, 1, 0, q>>>(&result[0], static_cast<int>(inp));
    sinfKernel<<<1, 1, 0, q>>>(&result[1], static_cast<int>(inp));

    double resultHost[N];
    cudaMemcpyAsync(resultHost, result, N * sizeof(double), cudaMemcpyDeviceToHost, q);

    cudaStreamSynchronize(q);

    auto const epsilon = std::numeric_limits<double>::epsilon();
    auto const epsilon_f = std::numeric_limits<float>::epsilon();
    REQUIRE_THAT(resultHost[0], Catch::Matchers::WithinAbs(sol, epsilon));
    REQUIRE_THAT(resultHost[1], Catch::Matchers::WithinAbs(sol, epsilon_f));
  }
  inputFile.close();
}
  cudaFreeAsync(result, q);
}
