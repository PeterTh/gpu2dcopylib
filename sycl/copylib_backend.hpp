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

  private:
	device_list devices;
	int64_t buffer_size;
};

bool is_2d_copy_available();

device_id execute_copy(executor& exec, const copy_spec& spec);

void execute_copy(executor& exec, const copy_plan& plan);

void execute_copy(executor& exec, const parallel_copy_set& set);

} // namespace copylib
