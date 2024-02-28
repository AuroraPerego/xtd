#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <sycl/sycl.hpp>
#include "math.h"

TEST_CASE("sinSycl", "[sin]") {
  constexpr int N = 4;
#ifdef ONEAPI_CPU
  auto queue =
      sycl::queue{sycl::cpu_selector_v, sycl::property::queue::in_order()};
#else
  auto queue =
      sycl::queue{sycl::gpu_selector_v, sycl::property::queue::in_order()};
#endif
  double *result = sycl::malloc_device<double>(N, queue);
{
  std::ifstream inputFile("Regr/sin/solutions.txt");
  double inp, sol;
  while (inputFile >> inp >> sol) {

    queue.submit([&](sycl::handler &cgh) {
      cgh.single_task([=]() {
        result[0] = static_cast<double>(xtd::sin(static_cast<float>(inp)));
        result[1] = static_cast<double>(xtd::sin(static_cast<double>(inp)));
        result[2] = static_cast<double>(xtd::sinf(static_cast<float>(inp)));
        result[3] = static_cast<double>(xtd::sinf(static_cast<double>(inp)));
      });
    });

    double resultHost[N];
    queue.memcpy(resultHost, result, N * sizeof(double));
    queue.wait();

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

    queue.submit([&](sycl::handler &cgh) {
      cgh.single_task([=]() {
        result[0] = static_cast<double>(xtd::sin(static_cast<int>(inp)));
        result[1] = static_cast<double>(xtd::sinf(static_cast<int>(inp)));
      });
    });

    double resultHost[N];
    queue.memcpy(resultHost, result, N * sizeof(double));
    queue.wait();

    auto const epsilon = std::numeric_limits<double>::epsilon();
    auto const epsilon_f = std::numeric_limits<float>::epsilon();
    REQUIRE_THAT(resultHost[0], Catch::Matchers::WithinAbs(sol, epsilon));
    REQUIRE_THAT(resultHost[1], Catch::Matchers::WithinAbs(sol, epsilon_f));
  }
  inputFile.close();
}
  sycl::free(result, queue);
}
