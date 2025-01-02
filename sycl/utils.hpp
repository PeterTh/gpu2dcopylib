#pragma once

#include <cassert> // IWYU pragma: keep (used in macro)
#include <format>  // IWYU pragma: keep (for std::format used in macro)
#include <functional>
#include <memory>          // IWYU pragma: keep (for std::hash)
#include <source_location> // IWYU pragma: keep (used in macro)
#include <string_view>

namespace copylib::utils {

template <class T>
void hash_combine(std::size_t& seed, const T& v) {
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <class T, class... Args>
size_t hash_args(const T& first, const Args&... args) {
	size_t seed = std::hash<T>{}(first);
	(hash_combine(seed, args), ...);
	return seed;
}

void print_to_cerr(std::string_view); // just to avoid including <iostream>

std::vector<std::string> split(const std::string& str, char delim);
} // namespace copylib::utils

#define COPYLIB_ENSURE(_expr, ...)                                                                                                                             \
	do {                                                                                                                                                       \
		const auto loc = std::source_location::current();                                                                                                      \
		if(!(_expr)) {                                                                                                                                         \
			copylib::utils::print_to_cerr(                                                                                                                     \
			    std::format("Error: !{}\nIn {}:{} : {}\n=> {}", #_expr, loc.file_name(), loc.line(), loc.function_name(), std::format(__VA_ARGS__)));          \
			assert(false);                                                                                                                                     \
			std::exit(1);                                                                                                                                      \
			__builtin_unreachable();                                                                                                                           \
		}                                                                                                                                                      \
	} while(false);


#define COPYLIB_ERROR(...) COPYLIB_ENSURE(false, __VA_ARGS__)

// Intel SYCL does not allow variadic function calls in device code
// this is generally just debug/informative output, so we can just disable it
#ifdef __INTEL_LLVM_COMPILER
#define COPYLIB_KERNEL_DEBUG_PRINTF(...)
#else
#define COPYLIB_KERNEL_DEBUG_PRINTF(...) printf(__VA_ARGS__)
#endif
