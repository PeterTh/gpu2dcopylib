#include "copylib.hpp" // IWYU pragma: keep

#include <chrono>
#include <vector>

using namespace copylib;

int main(int, char**) {
	constexpr int64_t buffer_size = 1024 * 1024 * 1024;
	constexpr int64_t queues_per_device = 2;
	executor exec(buffer_size, 2, queues_per_device);
	utils::print(exec.get_info());

	auto src_buffer = reinterpret_cast<intptr_t>(exec.get_buffer(device_id::d0));
	auto trg_buffer = reinterpret_cast<intptr_t>(exec.get_buffer(device_id::d1));

	const data_layout source_layout{src_buffer, 0, 8, 8 * 1024 * 1024, 128};
	const data_layout target_layout{trg_buffer, source_layout};

	COPYLIB_ENSURE(source_layout.total_extent() <= buffer_size, "Buffer too small for source layout");
	COPYLIB_ENSURE(target_layout.total_extent() <= buffer_size, "Buffer too small for target layout");

	const copy_spec spec{device_id::d0, source_layout, device_id::d1, target_layout};
	COPYLIB_ENSURE(is_valid(spec), "Invalid copy spec: {}", spec);

	auto q = exec.get_queue(device_id::d0);
	q.fill(reinterpret_cast<uint8_t*>(src_buffer), static_cast<uint8_t>(42), source_layout.total_extent()).wait_and_throw();

	using clock = std::chrono::high_resolution_clock;
	using namespace std::chrono_literals;
	std::vector<int64_t> chunk_sizes = {
	    0, 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024, 64 * 1024 * 1024};
	std::vector<std::vector<std::chrono::high_resolution_clock::duration>> durations(chunk_sizes.size());
	std::vector<copy_strategy> strategies;
	std::vector<parallel_copy_set> copy_sets;
	for(auto chunk : chunk_sizes) {
		const auto& strat = strategies.emplace_back(copy_type::staged, copy_properties::use_kernel, d2d_implementation::host_staging_at_source, chunk);
		const auto& set = copy_sets.emplace_back(manifest_strategy(spec, strat, basic_staging_provider{}));
		COPYLIB_ENSURE(is_equivalent(set, spec), "Copy set generated does not implement spec:\nspec:{}\nset:{}\n", spec, set);
	}

	utils::print("Copying {} MB between devices, strided on both ends in a buffer of {} MB\n", //
	    source_layout.total_bytes() / 1024 / 1024, source_layout.total_extent() / 1024 / 1024);

	constexpr int64_t repetitions = 200;

	for(size_t p = 0; p < chunk_sizes.size(); p++) {
		for(int64_t i = 0; i < repetitions; i++) {
			exec.barrier();
			auto start = clock::now();
			execute_copy(exec, copy_sets[p]);
			exec.barrier();
			auto end = clock::now();
			durations[p].push_back(end - start);
		}
	}

	for(size_t p = 0; p < chunk_sizes.size(); p++) {
		const auto min_time = utils::vector_min(durations[p]);
		using namespace std::chrono_literals;
		const auto time_seconds = min_time / 1.0s;
		const auto total_bytes = spec.source_layout.total_bytes();
		const auto total_gigabytes = total_bytes / (1024.0 * 1024.0 * 1024.0);
		const auto gigabytes_per_second = total_gigabytes / time_seconds;
		const std::string chunk_str = chunk_sizes[p] == 0 ? "no chunking" : std::to_string(chunk_sizes[p] / 1024 / 1024) + "MB chunks";
		utils::print("{:14}: {:10.2f}us, {:10.2f} GB/s ({})\n", chunk_str, time_seconds * 1e6, gigabytes_per_second, strategies[p]);
	}
}
