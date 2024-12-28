#pragma once

#include "copylib_core.hpp"

namespace copylib {

class executor {
  public:
	executor(int64_t buffer_size);
	sycl::queue& get_queue(device_id id);
	std::byte* get_buffer(device_id id);

  private:
	device_list devices;
};

void execute_copy(executor& exec, const copy_spec& spec);

bool is_2d_copy_available();

} // namespace copylib
