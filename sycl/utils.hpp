#pragma once

#include <algorithm>
#include <functional>
#include <string_view>
#include <vector>

#include <cassert>         // IWYU pragma: keep (used in macro)
#include <format>          // IWYU pragma: keep (for std::format used in macro)
#include <source_location> // IWYU pragma: keep (used in macro)

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

// just to avoid including <iostream>
void dump_to_cerr(std::string_view);
void dump_to_cout(std::string_view);

template <typename... Args>
void err_print(std::format_string<Args...> fmt, Args&&... args) {
	dump_to_cerr(std::format(fmt, std::forward<Args>(args)...));
}
template <typename... Args>
void print(std::format_string<Args...> fmt, Args&&... args) {
	dump_to_cout(std::format(fmt, std::forward<Args>(args)...));
}
inline void print(std::string_view str) { dump_to_cout(str); }

std::vector<std::string> split(const std::string& str, char delim);

template <typename T>
T vector_median(const std::vector<T>& values) {
	std::vector<T> sorted = values;
	std::sort(sorted.begin(), sorted.end());
	if(sorted.size() % 2 == 0) {
		return (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2;
	} else {
		return sorted[sorted.size() / 2];
	}
}

template <typename T>
T vector_min(const std::vector<T>& values) {
	return *std::min_element(values.begin(), values.end());
}

} // namespace copylib::utils

#define COPYLIB_ENSURE(_expr, ...)                                                                                                                             \
	do {                                                                                                                                                       \
		const auto loc = std::source_location::current();                                                                                                      \
		if(!(_expr)) {                                                                                                                                         \
			copylib::utils::err_print(                                                                                                                         \
			    "Error: !{}\nIn {}:{} : {}\n => {}\n", #_expr, loc.file_name(), loc.line(), loc.function_name(), std::format(__VA_ARGS__));                    \
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
