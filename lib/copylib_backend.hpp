#pragma once

#include "copylib_core.hpp"

namespace copylib {

struct device {
	sycl::device dev;
	std::vector<sycl::queue> queues;
	std::byte* dev_buffer = nullptr;
	std::byte* staging_buffer = nullptr;
	std::byte* host_buffer = nullptr;
	std::byte* host_staging_buffer = nullptr;

	~device();
};

using device_list = std::vector<device>;

class executor {
  public:
	struct target {
		device_id did;
		int64_t queue_idx;

		bool operator==(const target& other) const = default;
		bool operator!=(const target& other) const = default;
	};
	static constexpr target null_target = target{device_id::count, 0};

	executor(int64_t buffer_size);
	executor(int64_t buffer_size, int64_t devices_needed, int64_t queues_per_device = 1);

	sycl::queue& get_queue(device_id id, int64_t queue_idx = 0);
	sycl::queue& get_queue(const target& tgt) { return get_queue(tgt.did, tgt.queue_idx); }

	std::byte* get_buffer(device_id id);
	std::byte* get_staging_buffer(device_id id);
	std::byte* get_host_buffer(device_id id);
	std::byte* get_host_staging_buffer(device_id id);

	int64_t get_buffer_size() const { return buffer_size; }
	int64_t get_queues_per_device() const { return devices.front().queues.size(); }

	std::string get_sycl_impl_name() const;
	bool is_2d_copy_available() const;
	bool is_device_to_device_copy_available() const;
	bool is_peer_memory_access_available() const;
	int32_t get_preferred_wg_size() const;
	std::string get_info() const;

	enum class possibility {
		possible,
		needs_2d_copy,
		needs_d2d_copy,
	};

	possibility can_copy(const copy_spec& spec) const;
	possibility can_copy(const parallel_copy_set& spec) const;

	void barrier();

  private:
	mutable device_list devices; // Mutable due to ext_oneapi_can_access_peer not being const; very ugly
	std::vector<sycl::device> gpu_devices;
	int64_t buffer_size;
};


executor::target execute_copy(executor& exec, const copy_spec& spec, int64_t queue_idx = 0, const executor::target last_target = executor::null_target);

void execute_copy(executor& exec, const copy_plan& plan);

void execute_copy(executor& exec, const parallel_copy_set& set);

} // namespace copylib
