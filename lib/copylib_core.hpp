#pragma once

#include "utils.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>

namespace copylib {

enum class device_id : int16_t {
	host = -1,
	d0 = 0,
	d1 = 1,
	d2 = 2,
	d3 = 3,
	d4 = 4,
	d5 = 5,
	d6 = 6,
	d7 = 7,
	count = 8,
};

#pragma pack(push, 0)
struct staging_id {
	constexpr static uint8_t staging_id_flag = 0b00000001;
	uint8_t is_staging_id = staging_id_flag;
	uint8_t on_host = false;
	device_id did = device_id::d0;
	uint32_t index = 0;

	staging_id() = default;
	staging_id(bool on_host, device_id did, uint32_t index) : on_host(on_host), did(did), index(index) {}

	constexpr bool operator==(const staging_id& other) const = default;
	constexpr bool operator!=(const staging_id& other) const = default;
};
#pragma pack(pop)
static_assert(sizeof(staging_id) == sizeof(intptr_t));
static_assert(offsetof(staging_id, is_staging_id) == 0);

// data layout used as the source or destination of a copy operation
struct data_layout {
	union {
		intptr_t base = 0;
		staging_id staging;
	};
	int64_t offset = 0;
	int64_t fragment_length = 0;
	int64_t fragment_count = 1;
	int64_t stride = 0;

	constexpr data_layout() {}
	constexpr data_layout(intptr_t base, int64_t offset, int64_t fragment_length)
	    : base(base), offset(offset), fragment_length(fragment_length), fragment_count(1), stride(fragment_length) {}
	constexpr data_layout(intptr_t base, int64_t offset, int64_t fragment_length, int64_t fragment_count, int64_t stride)
	    : base(base), offset(offset), fragment_length(fragment_length), fragment_count(fragment_count), stride(stride) {}
	constexpr data_layout(intptr_t base, const data_layout& layout)
	    : base(base), offset(layout.offset), fragment_length(layout.fragment_length), fragment_count(layout.fragment_count), stride(layout.stride) {}

	data_layout(staging_id staging, int64_t offset, int64_t fragment_length)
	    : staging(staging), offset(offset), fragment_length(fragment_length), stride(fragment_length) {}
	data_layout(staging_id staging, int64_t offset, int64_t fragment_length, int64_t fragment_count, int64_t stride)
	    : staging(staging), offset(offset), fragment_length(fragment_length), fragment_count(fragment_count), stride(stride) {}

	constexpr int64_t total_bytes() const { return fragment_count * fragment_length; }
	constexpr int64_t total_extent() const { return offset + fragment_count * stride; }
	constexpr int64_t effective_stride() const { return stride == 0 ? fragment_length : stride; }
	constexpr bool unit_stride() const { return fragment_length == stride || (fragment_count == 1 && stride == 0); }
	constexpr int64_t fragment_offset(int64_t fragment) const {
		COPYLIB_ENSURE(fragment >= 0 && fragment < fragment_count, "Invalid fragment index (#{} of {} total)", fragment, fragment_count);
		return offset + fragment * stride;
	}
	constexpr int64_t end_offset() const { return fragment_offset(fragment_count - 1) + fragment_length; }

	constexpr bool is_unplaced_staging() const { return staging.is_staging_id == staging_id::staging_id_flag; }

	std::byte* base_ptr() const {
		COPYLIB_ENSURE(!is_unplaced_staging(), "Invalid base pointer (uninitialized staging?): {}", base);
		return reinterpret_cast<std::byte*>(base);
	}

	constexpr bool operator==(const data_layout& other) const {
		return base == other.base && offset == other.offset && fragment_length == other.fragment_length && fragment_count == other.fragment_count
		       && stride == other.stride;
	}
	constexpr bool operator!=(const data_layout& other) const { return !(*this == other); }
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

	constexpr copy_spec(device_id src_dev, const data_layout& src_layout, device_id tgt_dev, const data_layout& tgt_layout)
	    : source_device(src_dev), source_layout(src_layout), target_device(tgt_dev), target_layout(tgt_layout) {}
	constexpr copy_spec(device_id src_dev, const data_layout& src_layout, device_id tgt_dev, const data_layout& tgt_layout, copy_properties p)
	    : source_device(src_dev), source_layout(src_layout), target_device(tgt_dev), target_layout(tgt_layout), properties(p) {}

	[[nodiscard]] constexpr bool is_contiguous() const { return source_layout.unit_stride() && target_layout.unit_stride(); }
	[[nodiscard]] constexpr copy_spec with_properties(copy_properties p) const { return {source_device, source_layout, target_device, target_layout, p}; }

	constexpr bool operator==(const copy_spec&) const = default;
	constexpr bool operator!=(const copy_spec&) const = default;
};

// a copy plan is a list of one or more copy specifications which need to be enacted subsequently to implement one semantic copy operation
using copy_plan = std::vector<copy_spec>;

// a parallel copy set is a set of independent copy plans which can be enacted concurrently
using parallel_copy_set = std::vector<copy_plan>;

// defines the strategy type used to copy data between memories
enum class copy_type {
	direct, // copy directly from source to destination using copy operations
	staged, // stage/unstage to a linearized buffer to perform the copy
};

// defines how to deal with device to device copy operations
enum class d2d_implementation {
	direct,                 // directly copy from device to device
	host_staging_at_source, // stage in host memory at the source device
	host_staging_at_target, // stage in host memory at the target device
	host_staging_at_both,   // stage in host memory at both devices, with extra copy operation
};

// defines the strategy used to copy data between memories
struct copy_strategy {
	copy_type type = copy_type::direct;
	copy_properties properties = copy_properties::none;
	d2d_implementation d2d = d2d_implementation::direct;
	int64_t chunk_size = 0; // the size of each chunk to split the copy into, in bytes; 0 means no chunking

	copy_strategy() = default;
	copy_strategy(copy_type t) : type(t) {}
	copy_strategy(int64_t c) : chunk_size(c) {}
	copy_strategy(copy_type t, copy_properties p) : type(t), properties(p) {}
	copy_strategy(copy_type t, copy_properties p, d2d_implementation d) : type(t), properties(p), d2d(d) {}
	copy_strategy(copy_type t, copy_properties p, int64_t c) : type(t), properties(p), chunk_size(c) {}
	copy_strategy(copy_type t, copy_properties p, d2d_implementation d, int64_t c) : type(t), properties(p), d2d(d), chunk_size(c) {}

	constexpr bool operator==(const copy_strategy&) const = default;
	constexpr bool operator!=(const copy_strategy&) const = default;
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

using staging_buffer_provider = std::function<staging_id(device_id, bool, int64_t)>;

class basic_staging_provider {
  public:
	staging_id operator()(device_id did, bool on_host, int64_t size) {
		COPYLIB_ENSURE(size > 0, "Invalid staging buffer size: {}", size);
		COPYLIB_ENSURE(did != device_id::host, "Invalid staging buffer request: device id is host");
		return {on_host, did, next_staging_idx++};
	}

  private:
	uint32_t next_staging_idx = 0;
};

// apply staging to the given spec if requested by the strategy
copy_plan apply_staging(const copy_spec&, const copy_strategy&, const staging_buffer_provider&);

// apply staging to each copy spec in the given parallel copy set if requested by the strategy
parallel_copy_set apply_staging(const parallel_copy_set&, const copy_strategy&, const staging_buffer_provider&);

// apply the desired d2d implementation to the given copy plan
copy_plan apply_d2d_implementation(const copy_plan&, const d2d_implementation, const staging_buffer_provider&);

// apply the desired d2d implementation to the given parallel copy set (by applying it to each copy plan)
parallel_copy_set apply_d2d_implementation(const parallel_copy_set&, const d2d_implementation, const staging_buffer_provider&);

// manifests the copy strategy on the given copy spec, applying chunking and staging as necessary
parallel_copy_set manifest_strategy(const copy_spec&, const copy_strategy&, const staging_buffer_provider&);

} // namespace copylib
