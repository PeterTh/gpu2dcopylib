#include "copylib.hpp" // IWYU pragma: keep

#include <chrono>
#include <fstream>
#include <unordered_map>

#include <cmath>
#include <cstdio>
#include <sched.h>
#include <unistd.h>

using namespace copylib;

struct benchmark_layout {
	int64_t num_fragments;
	int64_t fragment_length;
	int64_t stride;
};

struct benchmark_device {
	device_id id;
	bool on_host;
	device_id get_exec_device() const { return on_host ? device_id::host : id; }
};

namespace d {
constexpr benchmark_device gpu(int64_t idx) { return {static_cast<device_id>(idx), false}; }
constexpr benchmark_device host(int64_t idx) { return {static_cast<device_id>(idx), true}; }
} // namespace d

struct benchmark_config {
	int64_t max_repetitions = 10;
	std::vector<std::pair<benchmark_device, benchmark_device>> device_pairs;
	std::vector<copy_type> types;
	std::vector<copy_properties> properties;
	std::vector<d2d_implementation> d2d_implementations;
	std::vector<int64_t> chunk_sizes;
	std::vector<benchmark_layout> layouts;
};

struct benchmark_spec {
	copy_spec spec;
	copy_strategy strat;

	constexpr bool operator==(const benchmark_spec&) const = default;
	constexpr bool operator!=(const benchmark_spec&) const = default;
};

namespace std {
template <>
struct hash<benchmark_spec> {
	size_t operator()(const benchmark_spec& p) const { return utils::hash_args(p.spec, p.strat); }
};
} // namespace std

template <typename T>
std::vector<std::pair<T, T>> generate_pairs(const std::vector<T>& values) {
	std::vector<std::pair<T, T>> pairs;
	for(const auto& a : values) {
		for(const auto& b : values) {
			pairs.push_back({a, b});
		}
	}
	return pairs;
}

int main(int, char**) {
	// create an executor with a buffer size of 1 GB
	constexpr int64_t buffer_size = 1024l * 1024l * 1024l * 1l;
	executor exec(buffer_size, 2, 2);
	constexpr int64_t max_copy_extent = buffer_size / 2;

	utils::print("Benchmark executor created:\n{}\n", exec.get_info());

	benchmark_config config = {
	    .device_pairs =
	        {
	            {d::gpu(0), d::host(0)},
	            {d::host(0), d::gpu(0)},
	            {d::gpu(0), d::gpu(1)},
	        },
	    .types = {copy_type::direct, copy_type::staged},
	    .properties = {copy_properties::none, copy_properties::use_kernel, copy_properties::use_2D_copy},
	    .d2d_implementations = {d2d_implementation::direct, d2d_implementation::host_staging_at_source, d2d_implementation::host_staging_at_target,
	        d2d_implementation::host_staging_at_both},
	    .chunk_sizes = {0, 1024, 2 * 1024, 8 * 1024},
	    .layouts = {},
	};

	// contiguous layouts up from 4 bytes to 512 MB
	for(int64_t i = 2; i < static_cast<int64_t>(log2(512 * 1024 * 1024)); i++) {
		const int64_t fragment_length = 1 << i;
		const int64_t stride = fragment_length;
		const int64_t num_fragments = 1;
		config.layouts.push_back({num_fragments, fragment_length, stride});
	}

	// 2D layouts with at most 32 MB total size, and at most 512*1024 fragments; fragment lengths from 4 bytes to 512 bytes
	for(int64_t i = 2; i < static_cast<int64_t>(log2(512)); i++) {
		const int64_t fragment_length = 1 << i;
		const int64_t stride = 1024; // 1 kB stride
		const int64_t num_fragments = std::min(std::min(32l * 1024l * 1024l / fragment_length, max_copy_extent / stride), 512l * 1024l);
		config.layouts.push_back({num_fragments, fragment_length, stride});
	}

	std::vector<benchmark_spec> benchmark_specs;
	for(const auto& device_pair : config.device_pairs) {
		for(const auto& type : config.types) {
			for(const auto& prop : config.properties) {
				for(const auto& d2d_impl : config.d2d_implementations) {
					for(const auto& chunk_size : config.chunk_sizes) {
						for(const auto& layout : config.layouts) {
							const auto src_dev = device_pair.first;
							const auto tgt_dev = device_pair.second;
							auto src_buffer = reinterpret_cast<intptr_t>(src_dev.on_host ? exec.get_host_buffer(src_dev.id) : exec.get_buffer(src_dev.id));
							auto tgt_buffer = reinterpret_cast<intptr_t>(tgt_dev.on_host ? exec.get_host_buffer(tgt_dev.id) : exec.get_buffer(tgt_dev.id));
							const copy_spec spec{
							    src_dev.get_exec_device(),
							    {src_buffer, 0, layout.fragment_length, layout.num_fragments, layout.stride},
							    tgt_dev.get_exec_device(),
							    {tgt_buffer, 0, layout.fragment_length, layout.num_fragments, layout.stride},
							};
							const copy_strategy strat{type, prop, d2d_impl, chunk_size};
							benchmark_specs.emplace_back(spec, strat);
						}
					}
				}
			}
		}
	}

	utils::print("Planned {} benchmarks with at most {} repetitions each\n", benchmark_specs.size(), config.max_repetitions);

	std::vector<std::pair<benchmark_spec, parallel_copy_set>> benchmarks;
	uint64_t removed_due_to_d2d = 0, removed_due_to_two_d = 0;
	// manifest all the plans and ensure they are valid and executable on the current executor
	int64_t manifested = 0;
	for(const auto& spec : benchmark_specs) {
		if(manifested % 10 == 0) {
			if(isatty(fileno(stdout))) {
				utils::print("\rManifesting benchmark {:7} / {:7} ({:5.1f}%)", manifested, benchmark_specs.size(), 100.0 * manifested / benchmark_specs.size());
			} else {
				utils::print(".");
			}
		}
		COPYLIB_ENSURE(spec.spec.source_layout.total_extent() <= exec.get_buffer_size(), "Source layout too large: {}", spec.spec.source_layout);
		COPYLIB_ENSURE(spec.spec.target_layout.total_extent() <= exec.get_buffer_size(), "Target layout too large: {}", spec.spec.target_layout);
		const auto copy_set = manifest_strategy(spec.spec, spec.strat, basic_staging_provider{});
		COPYLIB_ENSURE(is_valid(copy_set), "Invalid copy set: {}\n  -> generated for copy\n     {}\n     with strategy {}", copy_set, spec.spec, spec.strat);
		auto exec_possible = exec.can_copy(copy_set);
		if(exec_possible == executor::possibility::needs_d2d_copy) {
			removed_due_to_d2d++;
		} else if(exec_possible == executor::possibility::needs_2d_copy) {
			removed_due_to_two_d++;
		} else {
			benchmarks.emplace_back(spec, copy_set);
		}
		manifested++;
	}

	utils::print("\nWill perform {} benchmarks ({} removed due to d2d, {} removed due to 2d)\n", benchmarks.size(), removed_due_to_d2d, removed_due_to_two_d);

	char hostname[256];
	gethostname(hostname, 256);
	const auto timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	const auto log_filename = std::format("benchmark_{}_{}.log", hostname, timestamp);
	std::ofstream log(log_filename);
	log << exec.get_info() << std::endl << std::endl;

	using clk = std::chrono::high_resolution_clock;
	using namespace std::chrono_literals;
	const clk::duration max_time_for_config = 1s;

	std::unordered_map<benchmark_spec, std::vector<clk::duration>> results;
	std::unordered_map<benchmark_spec, std::pair<clk::duration, bool>> skipping_info;
	int64_t completed = 0;
	int64_t reporting_threshold = 10;
	int64_t total = benchmarks.size() * config.max_repetitions;
	for(int64_t i = 0; i < config.max_repetitions; i++) {
		for(const auto& [spec, copy_set] : benchmarks) {
			if(skipping_info[spec].first > max_time_for_config) {
				if(!skipping_info[spec].second) {
					log << std::format("Skipping run {} (and all subsequent) for {} / {} due to previous runs taking too long\n", i, spec.spec, spec.strat);
					skipping_info[spec].second = true;
				}
				continue;
			}
			exec.barrier();
			// if(i == 0) log << spec.spec << " ==> " << copy_set << std::endl;
			auto start = clk::now();
			execute_copy(exec, copy_set);
			exec.barrier();
			auto end = clk::now();
			results[spec].push_back(end - start);
			skipping_info[spec].first += end - start;
			completed++;
			if(isatty(fileno(stdout))) {
				if(completed % reporting_threshold == 0) {
					utils::print("\rCompleted {:9} / {:9} runs ({:5.1f}%)", completed, total, 100.0 * completed / total);
				}
			} else {
				// print a dot for every percent
				if(completed % (total / 100) == 0) { utils::print("."); }
			}
		}
	}
	utils::print("\n");
	log.close();

	std::unordered_map<benchmark_spec, double> median_times, median_gigabytes_per_second, mean_times, time_stddevs, times_25_percent, times_75_percent;
	for(const auto& [spec, durations] : results) {
		const auto metrics = utils::vector_metrics(durations);
		const auto time_seconds = metrics.median / 1.0s;
		median_times[spec] = time_seconds;
		times_25_percent[spec] = metrics.percentile_25 / 1.0s;
		times_75_percent[spec] = metrics.percentile_75 / 1.0s;
		const auto total_bytes = spec.spec.source_layout.total_bytes();
		const auto total_gigabytes = total_bytes / (1024.0 * 1024.0 * 1024.0);
		median_gigabytes_per_second[spec] = total_gigabytes / time_seconds;
		const auto mean = std::accumulate(durations.begin(), durations.end(), clk::duration(0)) / durations.size();
		const auto mean_seconds = mean / 1.0s;
		mean_times[spec] = mean_seconds;
		const auto time_variance = std::accumulate(durations.begin(), durations.end(), 0.0, [&](double acc, const auto& dur) {
			const auto dur_seconds = dur / 1.0s;
			return acc + (dur_seconds - mean_seconds) * (dur_seconds - mean_seconds);
		});
		time_stddevs[spec] = std::sqrt(time_variance / durations.size());
	}

	const auto output_filename = std::format("benchmark_results_{}_{}.csv", hostname, timestamp);
	std::fstream out(output_filename, std::ios::out | std::ios::trunc);
	out << "source_device,target_device,copy_type,copy_properties,d2d_implementation,chunk_size,num_fragments,fragment_length,stride,"
	       "median_time,time_25_percent,time_75_percent,mean_time,time_stddev,gigabytes_per_second\n";
	for(const auto& [bench, _] : benchmarks) {
		const auto& spec = bench.spec;
		const auto& strat = bench.strat;
		const auto& layout = spec.source_layout;
		const auto& median_time = median_times[bench];
		const auto& time_stddev = time_stddevs[bench];
		const auto& mean_time = mean_times[bench];
		const auto& time_25_percent = times_25_percent[bench];
		const auto& time_75_percent = times_75_percent[bench];
		const auto& gigs_per_second = median_gigabytes_per_second[bench];
		out << std::format("{:4},{:4}", spec.source_device, spec.target_device)
		    << std::format(",{:6},{:12},{:23},{:12}", strat.type, strat.properties, strat.d2d, strat.chunk_size)
		    << std::format(",{:12},{:12},{:12}", layout.fragment_count, layout.fragment_length, layout.stride)
		    << std::format(",{:12.6f},{:12.6f},{:12.6f},{:12.6f},{:12.6f},{:12.6f}\n", median_time, time_25_percent, time_75_percent, mean_time, time_stddev,
		           gigs_per_second);
	}
}