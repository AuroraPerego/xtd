#pragma once

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

//// Boost headers
// #define BOOST_STACKTRACE_USE_BACKTRACE
// #include <boost/stacktrace.hpp>

// HIP headers
#include <hip/hip_runtime.h>

namespace errorCheck {

[[noreturn]] inline void abortOnHipError(const char *file, int line,
                                          const char *cmd, const char *error,
                                          const char *message,
                                          const char *description = nullptr) {
  std::ostringstream out;
  out << "\n";
  out << file << ", line " << line << ":\n";
  out << "hipCheck(" << cmd << ");\n";
  out << error << ": " << message << "\n";
  if (description)
    out << description << "\n";

//  out << "\nCurrent stack trace:\n";
//  out << boost::stacktrace::stacktrace();
  out << "\n";

  throw std::runtime_error(out.str());
}

inline bool hipCheck_(const char *file, int line, const char *cmd,
                      hipError_t result, const char *description = nullptr) {
  if (result == hipSuccess)
    return true;

  const char *error = hipGetErrorName(result);
  const char *message = hipGetErrorString(result);
  abortOnHipError(file, line, cmd, error, message, description);
  return false;
}

} // namespace errorCheck

#define HIP_CHECK(ARG, ...)                                                    \
  (errorCheck::hipCheck_(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))
