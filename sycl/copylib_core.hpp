#pragma once

#include "utils.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <unordered_set>

namespace copylib {

struct device {
	sycl::queue q;
	std::byte* dev_buffer = nullptr;
	std::byte* staging_buffer = nullptr;
	std::byte* host_buffer = nullptr;
	std::byte* host_staging_buffer = nullptr;
	std::chrono::duration<double> linear_h_to_d_time{0};
	std::chrono::duration<double> linear_d_to_h_time{0};
};

using devices = std::vector<device>;
inline devices g_devices;

enum class device_id : int64_t {
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

using staging_buffer_provider = std::function<std::byte*(device_id, int64_t)>;

// data layout used as the source or destination of a copy operation
struct data_layout {
	std::byte* base = nullptr;
	int64_t offset = 0;
	int64_t fragment_length = 0;
	int64_t fragment_count = 1;
	int64_t stride = 0;

	constexpr int64_t total_bytes() const { return fragment_count * fragment_length; }
	constexpr int64_t total_extent() const { return offset + fragment_count * stride; }
	constexpr bool unit_stride() const { return fragment_length == stride || (fragment_count == 1 && stride == 0); }
	constexpr int64_t fragment_offset(int64_t fragment) const {
		COPYLIB_ENSURE(fragment >= 0 && fragment < fragment_count, "Invalid fragment index (#{} of {} total)", fragment, fragment_count);
		return offset + fragment * stride;
	}

	constexpr bool operator==(const data_layout& other) const = default;
	constexpr bool operator!=(const data_layout& other) const = default;
};

enum class copy_properties {
	none = 0x0000,
	use_kernel = 0x0001,  // whether to use a kernel to perform the copy
	use_2D_copy = 0x0010, // whether to use a native 2D copy operation, if available
};
inline copy_properties operator|(copy_properties a, copy_properties b) { return static_cast<copy_properties>(static_cast<int>(a) | static_cast<int>(b)); }
inline bool operator&(copy_properties a, copy_properties b) { return static_cast<int>(a) & static_cast<int>(b); }

// a copy specification describes a single copy operation from a source data layout and device to a destination data layout and device
struct copy_spec {
	device_id source_device;
	data_layout source_layout;
	device_id target_device;
	data_layout target_layout;

	copy_properties properties = copy_properties::none;

	constexpr bool is_contiguous() const { return source_layout.unit_stride() && target_layout.unit_stride(); }

	constexpr bool operator==(const copy_spec& other) const = default;
	constexpr bool operator!=(const copy_spec& other) const = default;
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
using copy_plan = std::vector<copy_spec>;

// a parallel copy set is a set of independent copy plans which can be enacted concurrently
using parallel_copy_set = std::unordered_set<copy_plan>;

// defines the strategy type used to copy data between memories
enum class copy_type {
	direct, // copy directly from source to destination using copy operations
	staged, // stage/unstage to a linearized buffer to perform the copy
};

// defines the strategy used to copy data between memories
struct copy_strategy {
	copy_type type = copy_type::direct;
	copy_properties properties = copy_properties::none;
	int64_t chunk_size = 0; // the size of each chunk to split the copy into, in bytes; 0 means no chunking
};

// validate whether a given data layout is sound
bool is_valid(const data_layout& layout);

// validate whether a given copy spec is sound
bool is_valid(const copy_spec& plan);

// check whether a given copy plan implements a given copy specification
bool is_equivalent(const copy_plan& plan, const copy_spec& spec);

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

// turn unit stride (contiguous) multi-fragment layouts into single fragment layouts
data_layout normalize(const data_layout&);

// turn contiguous multi-fragment copy specs into single fragment copy specs
copy_spec normalize(const copy_spec&);

// apply chunking to `spec` if requested by `strategy`
parallel_copy_set apply_chunking(const copy_spec&, const copy_strategy&);

// apply staging to `spec` if requested by `strategy`
copy_plan apply_staging(const copy_spec&, const copy_strategy&, const staging_buffer_provider&);

// apply staging to each copy spec in `copy_set` if requested by `strategy`
parallel_copy_set apply_staging(const parallel_copy_set&, const copy_strategy&, const staging_buffer_provider&);

} // namespace copylib
