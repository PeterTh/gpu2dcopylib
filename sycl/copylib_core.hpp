#pragma once

#include "utils.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <unordered_set>

namespace copylib {

struct device {
	sycl::queue queue;
	std::byte* dev_buffer = nullptr;
	std::byte* staging_buffer = nullptr;
	std::byte* host_buffer = nullptr;
	std::byte* host_staging_buffer = nullptr;
	std::chrono::duration<double> linear_h_to_d_time{0};
	std::chrono::duration<double> linear_d_to_h_time{0};

	~device();
};

using device_list = std::vector<device>;
inline device_list g_devices;

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

// data layout used as the source or destination of a copy operation
struct data_layout {
	intptr_t base = 0;
	int64_t offset = 0;
	int64_t fragment_length = 0;
	int64_t fragment_count = 1;
	int64_t stride = 0;

	constexpr int64_t total_bytes() const { return fragment_count * fragment_length; }
	constexpr int64_t total_extent() const { return offset + fragment_count * stride; }
	constexpr int64_t effective_stride() const { return stride == 0 ? fragment_length : stride; }
	constexpr bool unit_stride() const { return fragment_length == stride || (fragment_count == 1 && stride == 0); }
	constexpr int64_t fragment_offset(int64_t fragment) const {
		COPYLIB_ENSURE(fragment >= 0 && fragment < fragment_count, "Invalid fragment index (#{} of {} total)", fragment, fragment_count);
		return offset + fragment * stride;
	}
	constexpr int64_t end_offset() const { return fragment_offset(fragment_count - 1) + fragment_length; }

	static constexpr intptr_t min_staging_id = -100;
	constexpr bool is_unplaced_staging() const { return base < 0 && base > min_staging_id; }

	std::byte* base_ptr() const {
		COPYLIB_ENSURE(base > 0, "Invalid base pointer (uninitialized staging?): {}", base);
		return reinterpret_cast<std::byte*>(base);
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
	int64_t chunk_size = 0;        // the size of each chunk to split the copy into, in bytes; 0 means no chunking
	bool require_host_hop = false; // whether to require a host hop for the copy (if direct device <-> device transfer is not possible)
};

// validate whether a given data layout is sound
bool is_valid(const data_layout& layout);

// validate whether a given copy spec is sound
bool is_valid(const copy_spec& plan);

// validate whether a given copy plan is sound
bool is_valid(const copy_plan& plan);

// validate whether a given copy set is sound
bool is_valid(const parallel_copy_set& set);

// check whether a given copy plan implements a given copy specification
bool is_equivalent(const copy_plan& plan, const copy_spec& spec);

// check whetner the given copy set implements the given copy specification
bool is_equivalent(const parallel_copy_set& plan, const copy_spec& spec);

// turn unit stride (contiguous) multi-fragment layouts into single fragment layouts
data_layout normalize(const data_layout&);

// turn contiguous multi-fragment copy specs into single fragment copy specs
copy_spec normalize(const copy_spec&);

// apply given properties to the given copy spec
copy_spec apply_properties(const copy_spec&, const copy_properties&);

// apply chunking to the given copy spec if requested by the strategy
parallel_copy_set apply_chunking(const copy_spec&, const copy_strategy&);

using staging_buffer_provider = std::function<intptr_t(device_id, int64_t)>;

class basic_staging_provider {
  public:
	intptr_t operator()(device_id id, int64_t size) {
		COPYLIB_ENSURE(size > 0, "Invalid staging buffer size: {}", size);
		COPYLIB_ENSURE(next_staging_id > data_layout::min_staging_id, "Staging buffer overflow");
		return next_staging_id--;
	}

  private:
	int64_t next_staging_id = -1;
};

// apply staging to the given spec if requested by the strategy
copy_plan apply_staging(const copy_spec&, const copy_strategy&, const staging_buffer_provider&);

// apply staging to each copy spec in the given parallel copy set if requested by the strategy
parallel_copy_set apply_staging(const parallel_copy_set&, const copy_strategy&, const staging_buffer_provider&);

// manifests the copy strategy on the given copy spec, applying chunking and staging as necessary
parallel_copy_set manifest_strategy(const copy_spec&, const copy_strategy&, const staging_buffer_provider&);

} // namespace copylib
