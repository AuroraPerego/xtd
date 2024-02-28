#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <cmath>
#include <fstream>
#include "math.h"
#include <limits>

TEST_CASE("sinSerial", "[sin]") {
  auto const epsilon = std::numeric_limits<double>::epsilon();
  auto const epsilon_f = std::numeric_limits<float>::epsilon();
{
  std::ifstream inputFile("Regr/sin/solutions.txt");

  if (!inputFile.is_open()) {
        std::cerr << "Error opening file solutions.txt" << std::endl;
    }

  double inp, sol;
  while (inputFile >> inp >> sol) {
	REQUIRE_THAT(xtd::sin(static_cast<float>(inp)), Catch::Matchers::WithinAbs(sol, epsilon_f));
	REQUIRE_THAT(xtd::sin(static_cast<double>(inp)), Catch::Matchers::WithinAbs(sol, epsilon));
	REQUIRE_THAT(xtd::sinf(static_cast<float>(inp)), Catch::Matchers::WithinAbs(sol, epsilon_f));
	REQUIRE_THAT(xtd::sinf(static_cast<double>(inp)), Catch::Matchers::WithinAbs(sol, epsilon_f));
  }
  inputFile.close();
}
{
  std::ifstream inputFile("Regr/sin/solutionsInt.txt");

  if (!inputFile.is_open()) {
        std::cerr << "Error opening file solutions.txt" << std::endl;
    }

  double inp, sol;
  while (inputFile >> inp >> sol) {
	REQUIRE_THAT(xtd::sin(static_cast<int>(inp)), Catch::Matchers::WithinAbs(sol, epsilon));
	REQUIRE_THAT(xtd::sinf(static_cast<int>(inp)), Catch::Matchers::WithinAbs(sol, epsilon_f));
  }
  inputFile.close();
}
}
