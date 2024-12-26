#pragma once

#include <source_location>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <iostream>
#include <unordered_set>

template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

#define ensure(_expr, ...)                                                                                                                                     \
	do {                                                                                                                                                       \
		const auto loc = std::source_location::current();                                                                                                      \
		if(!(_expr)) {                                                                                                                                         \
			std::cerr << "Error: " << loc.file_name() << ":" << loc.line() << ": (" << loc.function_name() << ")\n" << std::format(__VA_ARGS__) << "\n";       \
			assert(false);                                                                                                                                     \
			std::exit(1);                                                                                                                                      \
			__builtin_unreachable();                                                                                                                           \
		}                                                                                                                                                      \
	} while(false);


#define error(...) ensure(false, __VA_ARGS__)

struct Device {
	sycl::queue q;
	std::byte* dev_buffer = nullptr;
	std::byte* staging_buffer = nullptr;
	std::byte* host_buffer = nullptr;
	std::byte* host_staging_buffer = nullptr;
	std::chrono::duration<double> linear_h_to_d_time{0};
	std::chrono::duration<double> linear_d_to_h_time{0};
};

inline std::vector<Device> g_devices;

enum class DeviceID : int64_t {
	host = -1,
	d0 = 0,
	d1 = 1,
	d2 = 2,
	d3 = 3,
	d4 = 4,
	d5 = 5,
	d6 = 6,
	d7 = 7,
};

// data layout used as the source or destination of a copy operation
struct DataLayout {
	std::byte* base = nullptr;
	int64_t offset = 0;
	int64_t fragment_length = 0;
	int64_t fragment_count = 0;
	int64_t stride = 0;

	constexpr int64_t total_bytes() const { return fragment_count * fragment_length; }
	constexpr int64_t total_extent() const { return offset + fragment_count * stride; }
	constexpr bool unit_stride() const { return fragment_length == stride; }
	constexpr int64_t fragment_offset(int64_t fragment) const {
		ensure(fragment >= 0 && fragment < fragment_count, "Invalid fragment index (#{} of {} total)", fragment, fragment_count);
		return offset + fragment * stride;
	}

	bool operator==(const DataLayout& other) const = default;
	bool operator!=(const DataLayout& other) const = default;
};

enum class CopyProperties {
	none = 0x0000,
	use_kernel = 0x0001,  // whether to use a kernel to perform the copy
	use_2D_copy = 0x0010, // whether to use a native 2D copy operation, if available
};
inline CopyProperties operator|(CopyProperties a, CopyProperties b) { return static_cast<CopyProperties>(static_cast<int>(a) | static_cast<int>(b)); }
inline bool operator&(CopyProperties a, CopyProperties b) { return static_cast<int>(a) & static_cast<int>(b); }

// a copy specification describes a single copy operation from a source data layout and device to a destination data layout and device
struct CopySpec {
	DeviceID source_device;
	DataLayout source_layout;
	DeviceID target_device;
	DataLayout target_layout;

	CopyProperties properties = CopyProperties::none;

	bool operator==(const CopySpec& other) const = default;
	bool operator!=(const CopySpec& other) const = default;
};

// get a device queue for the given copy specification
// sycl::queue& get_queue(const CopySpec& spec) {
// 	if(spec.source_device != DeviceID::host) {
// 		return g_devices[static_cast<int>(spec.source_device)].q;
// 	} else if(spec.target_device != DeviceID::host) {
// 		return g_devices[static_cast<int>(spec.target_device)].q;
// 	}
// 	error("Invalid copy specification");
// }

// a copy plan is a list of one or more copy specifications which need to be enacted subsequently to implement one semantic copy operation
using CopyPlan = std::vector<CopySpec>;

// a parallel copy set is a set of independent copy plans which can be enacted concurrently
using ParallelCopySet = std::unordered_set<CopyPlan>;


// make types hashable
namespace std {
template <>
struct hash<DataLayout> {
	std::size_t operator()(const DataLayout& layout) const {
		auto hash = std::hash<std::byte*>{}(layout.base);
		hash_combine(hash, layout.offset);
		hash_combine(hash, layout.fragment_length);
		hash_combine(hash, layout.fragment_count);
		hash_combine(hash, layout.stride);
		return hash;
	}
};
template <>
struct hash<CopySpec> {
	std::size_t operator()(const CopySpec& spec) const {
		auto hash = std::hash<DeviceID>{}(spec.source_device);
		hash_combine(hash, std::hash<DataLayout>{}(spec.source_layout));
		hash_combine(hash, std::hash<DeviceID>{}(spec.target_device));
		hash_combine(hash, std::hash<DataLayout>{}(spec.target_layout));
		return hash;
	}
};
template <>
struct hash<CopyPlan> {
	std::size_t operator()(const CopyPlan& plan) const {
		std::size_t hash = 0;
		for(const auto& spec : plan) {
			hash_combine(hash, std::hash<CopySpec>{}(spec));
		}
		return hash;
	}
};
} // namespace std

// make types format-printable (and ostream for Catch2)
template <>
struct std::formatter<DeviceID> : std::formatter<std::string> {
	auto format(const DeviceID& p, format_context& ctx) const {
		if(p == DeviceID::host) { return formatter<string>::format("host", ctx); }
		return formatter<string>::format(std::format("d{}", static_cast<int>(p)), ctx);
	}
};
template <>
struct std::formatter<DataLayout> : std::formatter<std::string> {
	auto format(const DataLayout& p, format_context& ctx) const {
		return formatter<string>::format(std::format("{{{}+{}, [{} * {}, {}]}}", //
		                                     static_cast<void*>(p.base), p.offset, p.fragment_length, p.fragment_count, p.stride),
		    ctx);
	}
};
template <>
struct std::formatter<CopySpec> : std::formatter<std::string> {
	auto format(const CopySpec& p, format_context& ctx) const {
		return formatter<string>::format(std::format("copy({}{}, {}{})", //
		                                     p.source_device, p.source_layout, p.target_device, p.target_layout),
		    ctx);
	}
};
namespace std {
std::ostream& operator<<(std::ostream& os, const DeviceID& p);
std::ostream& operator<<(std::ostream& os, const DataLayout& p);
std::ostream& operator<<(std::ostream& os, const CopySpec& p);
} // namespace std

// validate whether a given data layout is sound
bool is_valid(const DataLayout& layout);

// validate whether a given copy spec is sound
bool is_valid(const CopySpec& plan);

// check whether a given copy plan implements a given copy specification
bool is_equivalent(const CopyPlan& plan, const CopySpec& spec);

// defines the strategy type used to copy data between memories
enum class CopyType {
	direct, // copy directly from source to destination using copy operations
	staged, // stage/unstage to a linearized buffer to perform the copy
};

// defines the strategy used to copy data between memories
struct CopyStrategy {
	CopyType type = CopyType::direct;
	CopyProperties properties = CopyProperties::none;
	int64_t chunk_size = 0; // the size of each chunk to split the copy into, in bytes; 0 means no chunking
};

// sycl::event perform_1D_copy(const CopySpec& spec, const CopyStrategy& strategy) {
// 	ensure(spec.source_layout.unit_stride() && spec.target_layout.unit_stride());
// 	ensure(spec.source_layout.total_bytes() == spec.target_layout.total_bytes());
// 	if(strategy.chunk_size == 0) {
// 		// perform a single copy operation
// 		auto& q = get_queue(spec);
// 		const auto src = spec.source_layout.base + spec.source_layout.offset;
// 		const auto tgt = spec.target_layout.base + spec.target_layout.offset;
// 		return q.copy(src, tgt, spec.source_layout.total_bytes());
// 	} else {
// 		// split the copy into chunks
// 		const auto total_bytes = spec.source_layout.total_bytes();
// 		const auto num_chunks = (total_bytes + strategy.chunk_size - 1) / strategy.chunk_size;
// 		sycl::event ev;
// 		for(int64_t i = 0; i < num_chunks; i++) {
// 			const auto chunk_offset = i * strategy.chunk_size;
// 			const auto chunk_start_fragment = chunk_offset / spec.source_layout.fragment_length;
// 			auto& q = get_queue(spec);
// 			ev = q.copy(spec.target_layout.base + target_offset, spec.source_layout.base + source_offset, chunk_size);
// 		}
// 		return ev;
// 	}
// }

// sycl::event perform_2D_copy(const CopySpec& spec, const CopyStrategy& strategy) {
// 	if(strategy.type == CopyType::direct) {
// #if SYCL_EXT_ONEAPI_MEMCPY2D >= 1
// #endif
// 	} else {
// 		// perform a staged copy
// 		error("Staged copy not implemented");
// 	}
// }

// sycl::event perform_copy(const CopySpec& spec, const CopyStrategy& strategy) {
// 	if(spec.source_layout.unit_stride() && spec.target_layout.unit_stride()) {
// 		return perform_1D_copy(spec, strategy);
// 	} else {
// 		return perform_2D_copy(spec, strategy);
// 	}
// }

ParallelCopySet apply_chunking(const CopySpec& spec, const CopyStrategy& strategy);

ParallelCopySet apply_staging(const ParallelCopySet& spec, const CopyStrategy& strategy);
