#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "math.h"
#include <cmath>
#include <limits>
#include <sycl/sycl.hpp>

TEST_CASE("sinSycl", "[sin]") {
  constexpr int N = 6;
#ifdef ONEAPI_CPU
  auto queue =
      sycl::queue{sycl::cpu_selector_v, sycl::property::queue::in_order()};
#else
  auto queue =
      sycl::queue{sycl::gpu_selector_v, sycl::property::queue::in_order()};
#endif
  double *result = sycl::malloc_device<double>(N, queue);

  std::vector<double> values{-1., 0., M_PI / 2, M_PI, 42.};

  for (auto v : values) {
    queue.submit([&](sycl::handler &cgh) {
      cgh.single_task([=]() {
        result[0] = static_cast<double>(xtd::sin(static_cast<int>(v)));
        result[1] = static_cast<double>(xtd::sin(static_cast<float>(v)));
        result[2] = static_cast<double>(xtd::sin(static_cast<double>(v)));
        result[3] = static_cast<double>(xtd::sinf(static_cast<int>(v)));
        result[4] = static_cast<double>(xtd::sinf(static_cast<float>(v)));
        result[5] = static_cast<double>(xtd::sinf(static_cast<double>(v)));
      });
    });

    double resultHost[N];
    queue.memcpy(resultHost, result, N * sizeof(double));
    queue.wait();

    auto const epsilon = std::numeric_limits<double>::epsilon();
    auto const epsilon_f = std::numeric_limits<float>::epsilon();
    REQUIRE_THAT(resultHost[0], Catch::Matchers::WithinAbs(
                                    std::sin(static_cast<int>(v)), epsilon));
    REQUIRE_THAT(resultHost[1],
                 Catch::Matchers::WithinAbs(std::sin(v), epsilon_f));
    REQUIRE_THAT(resultHost[2],
                 Catch::Matchers::WithinAbs(std::sin(v), epsilon));
    REQUIRE_THAT(resultHost[3], Catch::Matchers::WithinAbs(
                                    sinf(static_cast<int>(v)), epsilon_f));
    REQUIRE_THAT(resultHost[4], Catch::Matchers::WithinAbs(sinf(v), epsilon_f));
    REQUIRE_THAT(resultHost[5], Catch::Matchers::WithinAbs(sinf(v), epsilon_f));
  }
  sycl::free(result, queue);
}
