#include "copylib.hpp" // IWYU pragma: keep

#include <chrono>
#include <vector>

using namespace copylib;

int main(int, char**) {
	constexpr int64_t buffer_size = 256 * 1024 * 1024;
	executor exec(buffer_size, 1);
	utils::print(exec.get_info());

	auto src_buffer = reinterpret_cast<intptr_t>(exec.get_buffer(device_id::d0));
	auto trg_buffer = reinterpret_cast<intptr_t>(exec.get_staging_buffer(device_id::d0));

	const data_layout strided_layout{src_buffer, 0, 4, 8192 * 4, 2048 * 4};
	const data_layout linear_layout{trg_buffer, 0, strided_layout.fragment_length, strided_layout.fragment_count, strided_layout.fragment_length};
	const data_layout source_layout = strided_layout;
	const data_layout target_layout = linear_layout;

	COPYLIB_ENSURE(source_layout.total_extent() <= buffer_size, "Buffer too small for source layout");
	COPYLIB_ENSURE(target_layout.total_extent() <= buffer_size, "Buffer too small for target layout");

	const copy_spec spec{device_id::d0, source_layout, device_id::d0, target_layout};
	COPYLIB_ENSURE(is_valid(spec), "Invalid copy spec: {}", spec);

	auto q = exec.get_queue(device_id::d0);
	q.fill(reinterpret_cast<uint8_t*>(src_buffer), static_cast<uint8_t>(42), source_layout.total_extent()).wait_and_throw();

	using clock = std::chrono::high_resolution_clock;
	using namespace std::chrono_literals;
	std::vector<copy_properties> prop_options = {copy_properties::none, copy_properties::use_kernel};
	if(exec.is_2d_copy_available()) { prop_options.push_back(copy_properties::use_2D_copy); }
	std::vector<std::vector<std::chrono::high_resolution_clock::duration>> durations(prop_options.size());

	constexpr int64_t repetitions = 500;
	constexpr int64_t runs = 50;

	for(int64_t run = 0; run < runs; ++run) {
		for(size_t p = 0; p < prop_options.size(); p++) {
			int64_t reps = (prop_options[p] == copy_properties::none) ? repetitions / 50 : repetitions;
			const auto props = prop_options[p];
			const auto cur_spec = spec.with_properties(props);
			COPYLIB_ENSURE(is_valid(cur_spec), "Invalid current copy spec: {}", cur_spec);
			COPYLIB_ENSURE(exec.can_copy(cur_spec) == executor::possibility::possible, "Cannot execute copy with spec");
			exec.barrier();
			auto start = clock::now();
			for(int64_t i = 0; i < reps; i++) {
				execute_copy(exec, cur_spec);
			}
			exec.barrier();
			auto end = clock::now();
			durations[p].push_back((end - start) / reps);
		}
	}

	for(size_t p = 0; p < prop_options.size(); p++) {
		const auto min_time = utils::vector_min(durations[p]);
		using namespace std::chrono_literals;
		const auto time_seconds = min_time / 1.0s;
		const auto total_bytes = spec.source_layout.total_bytes();
		const auto total_gigabytes = total_bytes / (1024.0 * 1024.0 * 1024.0);
		const auto gigabytes_per_second = total_gigabytes / time_seconds;
		utils::print("{:12}: {:10.2f}us, {:10.2f} GB/s\n", prop_options[p], time_seconds * 1e6, gigabytes_per_second);
	}
}
