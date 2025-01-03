#pragma once

#include "copylib_core.hpp"

namespace copylib {

class executor {
  public:
	executor(int64_t buffer_size);
	sycl::queue& get_queue(device_id id);
	std::byte* get_buffer(device_id id);
	std::byte* get_staging_buffer(device_id id);
	std::byte* get_host_buffer(device_id id);
	std::byte* get_host_staging_buffer(device_id id);

	int64_t get_buffer_size() const { return buffer_size; }

	std::string get_sycl_impl_name() const;
	bool is_2d_copy_available() const;
	bool is_device_to_device_copy_available() const;
	std::string get_info() const;

	enum class possibility {
		possible,
		needs_2d_copy,
		needs_d2d_copy,
	};

	possibility can_copy(const parallel_copy_set& spec) const;

	void barrier();

  private:
	device_list devices;
	std::vector<sycl::device> gpu_devices;
	int64_t buffer_size;
};


device_id execute_copy(executor& exec, const copy_spec& spec);

void execute_copy(executor& exec, const copy_plan& plan);

void execute_copy(executor& exec, const parallel_copy_set& set);

} // namespace copylib
