#include "copylib.hpp" // IWYU pragma: keep

#include <chrono>
#include <vector>

using namespace copylib;

int main(int, char**) {
	constexpr int64_t buffer_size = 128 * 1024 * 1024;
	executor exec(buffer_size);
	utils::print(exec.get_info());

	auto src_buffer = reinterpret_cast<intptr_t>(exec.get_buffer(device_id::d0));
	auto trg_buffer = reinterpret_cast<intptr_t>(exec.get_staging_buffer(device_id::d0));

	const data_layout source_layout{src_buffer, 0, 8, 8192, 2024 * 8};
	const data_layout target_layout{trg_buffer, 0, 8, 8192, 8};

	COPYLIB_ENSURE(source_layout.total_extent() <= buffer_size, "Buffer too small for source layout");
	COPYLIB_ENSURE(target_layout.total_extent() <= buffer_size, "Buffer too small for target layout");

	const copy_spec spec{device_id::d0, source_layout, device_id::d0, target_layout};
	COPYLIB_ENSURE(is_valid(spec), "Invalid copy spec: {}", spec);

	auto q = exec.get_queue(device_id::d0);
	q.fill(reinterpret_cast<uint8_t*>(src_buffer), static_cast<uint8_t>(42), source_layout.total_extent()).wait_and_throw();

	using clock = std::chrono::high_resolution_clock;
	using namespace std::chrono_literals;
	std::vector<copy_properties> prop_options = {/*copy_properties::none,*/ copy_properties::use_kernel};
	if(exec.is_2d_copy_available()) { prop_options.push_back(copy_properties::use_2D_copy); }
	std::vector<std::vector<std::chrono::high_resolution_clock::duration>> durations(prop_options.size());

	constexpr int64_t repetitions = 500;

	for(size_t p = 0; p < prop_options.size(); p++) {
		for(int64_t i = 0; i < repetitions; i++) {
			const auto props = prop_options[p];
			const auto cur_spec = spec.with_properties(props);
			COPYLIB_ENSURE(is_valid(cur_spec), "Invalid current copy spec: {}", cur_spec);
			COPYLIB_ENSURE(exec.can_copy(cur_spec) == executor::possibility::possible, "Cannot execute copy with spec");
			exec.barrier();
			auto start = clock::now();
			execute_copy(exec, cur_spec);
			exec.barrier();
			auto end = clock::now();
			durations[p].push_back(end - start);
		}
	}

	for(size_t p = 0; p < prop_options.size(); p++) {
		const auto median = utils::vector_median(durations[p]);
		using namespace std::chrono_literals;
		const auto time_seconds = median / 1.0s;
		const auto total_bytes = spec.source_layout.total_bytes();
		const auto total_gigabytes = total_bytes / (1024.0 * 1024.0 * 1024.0);
		const auto gigabytes_per_second = total_gigabytes / time_seconds;
		utils::print("{:12}: {:6.3f}s, {:6.3f} GB/s\n", prop_options[p], time_seconds, gigabytes_per_second);
	}
}
