#include "copylib.hpp" // IWYU pragma: keep

#include <chrono>
#include <vector>

using namespace copylib;
using utils::parse_command_line_option;

int main(int argc, char** argv) {
	const int64_t buffer_size = parse_command_line_option(argc, argv, "--buffer-size", 256 * 1024 * 1024);
	const int64_t queues_per_device = parse_command_line_option(argc, argv, "--queues-per-device", 2);
	executor exec(buffer_size, 2, queues_per_device);
	utils::print(exec.get_info());

	const copy_type c_type =
	    parse_command_line_option<copy_type>(argc, argv, "--type", {{"direct", copy_type::direct}, {"staged", copy_type::staged}}, copy_type::staged);
	const copy_properties c_props = parse_command_line_option<copy_properties>(argc, argv, "--props",
	    {{"none", copy_properties::none}, {"kernel", copy_properties::use_kernel}, {"2D", copy_properties::use_2D_copy}}, copy_properties::use_kernel);
	const d2d_implementation c_d2d = parse_command_line_option<d2d_implementation>(argc, argv, "--d2d-impl",
	    {{"direct", d2d_implementation::direct}, {"host-source", d2d_implementation::host_staging_at_source},
	        {"host-target", d2d_implementation::host_staging_at_target}, {"host-both", d2d_implementation::host_staging_at_both}},
	    d2d_implementation::host_staging_at_source);
	const auto frag_length = parse_command_line_option(argc, argv, "--frag-length", 4);
	const auto frag_count = parse_command_line_option(argc, argv, "--frag-count", 1024 * 1024 * 4 / 4);
	const auto stride = parse_command_line_option(argc, argv, "--stride", 1024);
	const int64_t repetitions = parse_command_line_option(argc, argv, "--reps", 10);

	auto src_buffer = reinterpret_cast<intptr_t>(exec.get_buffer(device_id::d0));
	auto trg_buffer = reinterpret_cast<intptr_t>(exec.get_buffer(device_id::d1));

	const data_layout source_layout{src_buffer, 0, frag_length, frag_count, stride};
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
	    0, 512 * 1024, 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024, 64 * 1024 * 1024};

	// drop chunk sizes larger than the total size in the layout
	chunk_sizes.erase(
	    std::remove_if(chunk_sizes.begin(), chunk_sizes.end(), [&](int64_t chunk) { return chunk > source_layout.total_bytes(); }), chunk_sizes.end());

	std::vector<std::vector<std::chrono::high_resolution_clock::duration>> durations(chunk_sizes.size());
	std::vector<copy_strategy> strategies;
	std::vector<parallel_copy_set> copy_sets;
	for(auto chunk : chunk_sizes) {
		const auto& strat = strategies.emplace_back(c_type, c_props, c_d2d, chunk);
		const auto& set = copy_sets.emplace_back(manifest_strategy(spec, strat, basic_staging_provider{}));
		COPYLIB_ENSURE(is_equivalent(set, spec), "Copy set generated does not implement spec:\nspec:{}\nset:{}\n", spec, set);
	}

	utils::print("Copying {} MB between devices, strided on both ends in a buffer of {} MB, {} repetitions\n", //
	    source_layout.total_bytes() / 1024 / 1024, source_layout.total_extent() / 1024 / 1024, repetitions);

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
