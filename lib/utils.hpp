#pragma once

#include <algorithm>
#include <cmath>
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


template <typename T, typename Ret = decltype(std::declval<T>() * 0.5)>
Ret vector_percentile(const std::vector<T>& values, double percentile, bool is_sorted = false) {
	std::vector<T> sorted = values;
	if(!is_sorted) std::sort(sorted.begin(), sorted.end());
	double div = (sorted.size() - 1) * percentile;
	double rem = div - std::floor(div);
	size_t idx = std::floor(div);
	if(rem >= 0.0001 && idx + 1 < sorted.size()) {
		return (sorted[idx] * (1.0 - rem) + sorted[idx + 1] * rem);
	} else {
		return sorted[idx] * 1.0;
	}
}

template <typename T, typename Ret = decltype(std::declval<T>() * 0.5)>
Ret vector_median(const std::vector<T>& values) {
	return vector_percentile(values, 0.5);
}

template <typename T>
struct metrics {
	T median;
	T percentile_25;
	T percentile_75;
};

template <typename T, typename Ret = decltype(std::declval<T>() * 0.5)>
metrics<Ret> vector_metrics(const std::vector<T>& values) {
	metrics<Ret> result;
	std::vector<T> sorted = values;
	std::sort(sorted.begin(), sorted.end());
	result.percentile_25 = vector_percentile(sorted, 0.25, true);
	result.median = vector_percentile(sorted, 0.5, true);
	result.percentile_75 = vector_percentile(sorted, 0.75, true);
	return result;
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
