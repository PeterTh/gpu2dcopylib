#include "copylib.hpp" // IWYU pragma: keep

#include <chrono>
#include <vector>

using namespace copylib;

int main(int, char**) {
	auto src_buffer = reinterpret_cast<intptr_t>(nullptr);
	auto trg_buffer = src_buffer;

	const data_layout source_layout{src_buffer, 0, 8, 512 * 1024, 128};
	const data_layout target_layout{trg_buffer, source_layout};

	const copy_spec spec{device_id::d0, source_layout, device_id::d1, target_layout};
	COPYLIB_ENSURE(is_valid(spec), "Invalid copy spec: {}", spec);

	using clock = std::chrono::high_resolution_clock;
	using namespace std::chrono_literals;

	std::vector<int64_t> chunk_sizes = {0, 256, 512, 1024};
	std::vector<std::vector<std::chrono::high_resolution_clock::duration>> durations(chunk_sizes.size());
	std::vector<copy_strategy> strategies;
	std::vector<parallel_copy_set> copy_sets;
	for(auto chunk : chunk_sizes) {
		strategies.emplace_back(copy_type::staged, copy_properties::use_kernel, d2d_implementation::host_staging_at_source, chunk);
	}

	constexpr int64_t repetitions = 10;

	for(int64_t i = 0; i < repetitions; i++) {
		for(size_t p = 0; p < chunk_sizes.size(); p++) {
			auto start = clock::now();
			auto set = manifest_strategy(spec, strategies[p], basic_staging_provider{});
			auto end = clock::now();
			durations[p].push_back(end - start);
			if(i == 0) copy_sets.push_back(set);
		}
	}

	for(size_t p = 0; p < chunk_sizes.size(); p++) {
		const auto median_time = utils::vector_median(durations[p]);
		using namespace std::chrono_literals;
		const auto time_seconds = median_time / 1.0s;
		utils::print("{:9d} chunks: {:10.2f}us\n", copy_sets[p].size(), time_seconds * 1e6);
	}
}
