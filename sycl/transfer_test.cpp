
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <format>
#include <sched.h>

#include <sycl/sycl.hpp>

#define ensure(expr)                                                                                                                                           \
	if(!(expr)) {                                                                                                                                              \
		std::cerr << "Error on line " << __LINE__ << ": " << #expr << std::endl;                                                                               \
		std::exit(1);                                                                                                                                          \
		__builtin_unreachable();                                                                                                                               \
	}

#define error(msg) ensure(false && msg)

struct Device {
	sycl::queue q;
	std::byte* dev_buffer = nullptr;
	std::byte* staging_buffer = nullptr;
	std::byte* host_buffer = nullptr;
	std::byte* host_staging_buffer = nullptr;
	std::chrono::duration<double> linear_h_to_d_time{0};
	std::chrono::duration<double> linear_d_to_h_time{0};
};

std::vector<Device> g_devices;

enum class DeviceID : int64_t {
	host = -1,
	d0 = 0,
	d1 = 1,
	d2 = 2,
	d3 = 3,
	d4 = 4,
	d5 = 5,
	d6 = 6,
	d7 = 7,
};

// data layout used as the source or destination of a copy operation
struct DataLayout {
	std::byte* base = nullptr;
	int64_t offset = 0;
	int64_t fragment_length = 0;
	int64_t fragment_count = 0;
	int64_t stride = 0;

	int64_t total_bytes() const { return fragment_count * fragment_length; }
	int64_t total_extent() const { return offset + fragment_count * stride; }
	bool unit_stride() const { return fragment_length == stride; }
	int64_t fragment_offset(int64_t fragment) const {
		ensure(fragment >= 0 && fragment < fragment_count);
		return offset + fragment * stride;
	}

	bool operator==(const DataLayout& other) const = default;
	bool operator!=(const DataLayout& other) const = default;
};

// a copy specification describes a single copy operation from a source data layout and device to a destination data layout and device
struct CopySpec {
	DeviceID source_device;
	DataLayout source_layout;
	DeviceID target_device;
	DataLayout target_layout;
};

// get a device queue for the given copy specification
sycl::queue& get_queue(const CopySpec& spec) {
	if(spec.source_device != DeviceID::host) {
		return g_devices[static_cast<int>(spec.source_device)].q;
	} else if(spec.target_device != DeviceID::host) {
		return g_devices[static_cast<int>(spec.target_device)].q;
	}
	error("Invalid copy specification");
}

// a copy plan is a list of one or more copy specifications which need to be enacted subsequently to implement one semantic copy operation
using CopyPlan = std::vector<CopySpec>;

// a parallel copy set is a set of independent copy plans which can be enacted concurrently
using ParallelCopySet = std::set<CopyPlan>;

// check whether a given copy plan implements a given copy specification
bool is_equivalent(const CopyPlan& plan, const CopySpec& spec) {
	if(plan.empty()) { return false; }
	const auto& first_spec = plan.front();
	const auto& last_spec = plan.back();
	return first_spec.source_device == spec.source_device && first_spec.source_layout == spec.source_layout && last_spec.target_device == spec.target_device
	       && last_spec.target_layout == spec.target_layout;
}

// defines the strategy type used to copy data between memories
enum class CopyType {
	direct,         // copy directly from source to destination using a copy operations (or sequence of operations if no 2D copy is available)
	staged_copies,  // stage/unstage to a linearized buffer to perform the copy, using copy operations
	staged_kernels, // stage/unstage to a linearized buffer to perform the copy, using kernels to perform the staging operations
};

// defines the strategy used to copy data between memories
struct CopyStrategy {
	CopyType type = CopyType::direct;
	int64_t chunk_size = 0; // the size of each chunk to split the copy into, in bytes; 0 means no chunking
};

sycl::event perform_1D_copy(const CopySpec& spec, const CopyStrategy& strategy) {
	ensure(spec.source_layout.unit_stride() && spec.target_layout.unit_stride());
	ensure(spec.source_layout.total_bytes() == spec.target_layout.total_bytes());
	if(strategy.chunk_size == 0) {
		// perform a single copy operation
		auto& q = get_queue(spec);
		const auto src = spec.source_layout.base + spec.source_layout.offset;
		const auto tgt = spec.target_layout.base + spec.target_layout.offset;
		return q.copy(src, tgt, spec.source_layout.total_bytes());
	} else {
		// split the copy into chunks
		const auto total_bytes = spec.source_layout.total_bytes();
		const auto num_chunks = (total_bytes + strategy.chunk_size - 1) / strategy.chunk_size;
		sycl::event ev;
		for(int64_t i = 0; i < num_chunks; i++) {
			const auto chunk_offset = i * strategy.chunk_size;
			const auto chunk_start_fragment = chunk_offset / spec.source_layout.fragment_length;
			auto& q = get_queue(spec);
			ev = q.copy(spec.target_layout.base + target_offset, spec.source_layout.base + source_offset, chunk_size);
		}
		return ev;
	}
}

ParallelCopySet apply_chunking(const CopySpec& spec, const CopyStrategy& strategy) {
	if(strategy.chunk_size == 0) { return {{spec}}; }
	ParallelCopySet copy_set;
	if(spec.source_layout.unit_stride() && spec.target_layout.unit_stride()) {
		// split the contiguous copy into chunks
		const auto total_bytes = spec.source_layout.total_bytes();
		const auto num_chunks = (total_bytes + strategy.chunk_size - 1) / strategy.chunk_size;
		for(int64_t i = 0; i < num_chunks; i++) {
			const auto start_offset = i * strategy.chunk_size;
			const auto source_offset = spec.source_layout.offset + start_offset;
			const auto target_offset = spec.target_layout.offset + start_offset;
			copy_set.insert({{//
			    spec.source_device, {spec.source_layout.base, source_offset, spec.source_layout.fragment_length, 1, spec.source_layout.stride},
			    spec.target_device, {spec.target_layout.base, target_offset, spec.target_layout.fragment_length, 1, spec.target_layout.stride}}});
		}
	}

	sycl::event perform_2D_copy(const CopySpec& spec, const CopyStrategy& strategy) {
		if(strategy.type == CopyType::direct) {
#if SYCL_EXT_ONEAPI_MEMCPY2D >= 1
#endif
		} else {
			// perform a staged copy
			error("Staged copy not implemented");
		}
	}

	sycl::event perform_copy(const CopySpec& spec, const CopyStrategy& strategy) {
		if(spec.source_layout.unit_stride() && spec.target_layout.unit_stride()) {
			return perform_1D_copy(spec, strategy);
		} else {
			return perform_2D_copy(spec, strategy);
		}
	}

	int main(int argc, char** argv) {
		constexpr int total_bytes = 1024 * 1024 * 1024; // 1 GB

		const int num_repeats = 5;
		const int num_warmups = 2;
		const int total_runs = num_repeats + num_warmups;

		const auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);

		// print device info
		{
			int i = 0;
			for(const auto& device : gpu_devices) {
				std::cout << std::format("Device {:2}: {}\n", i++, device.get_info<sycl::info::device::name>());
			}
		}

		cpu_set_t prior_mask;
		CPU_ZERO(&prior_mask);
		ensure(pthread_getaffinity_np(pthread_self(), sizeof(prior_mask), &prior_mask) == 0);

		// allocate queues and device buffers
		int dev_id = 0;
		for(const auto& device : gpu_devices) {
			auto& dev = g_devices.emplace_back(sycl::queue(device));
			dev.dev_buffer = sycl::malloc_device<std::byte>(total_bytes, dev.q);
			dev.staging_buffer = sycl::malloc_device<std::byte>(total_bytes, dev.q);
			ensure(dev.dev_buffer != nullptr);
			ensure(dev.staging_buffer != nullptr);

			cpu_set_t mask_for_device;
			CPU_ZERO(&mask_for_device);
			CPU_SET(32 * dev_id, &mask_for_device); // TODO fix hardcoded NUMA <-> device mapping
			ensure(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask_for_device) == 0);

			dev.host_buffer = sycl::malloc_host<std::byte>(total_bytes, dev.q);
			dev.host_staging_buffer = sycl::malloc_host<std::byte>(total_bytes, dev.q);
			ensure(dev.host_buffer != nullptr);
			// initialize data on host
			for(int i = 0; i < total_bytes; i++) {
				dev.host_buffer[i] = static_cast<std::byte>(i % 256);
				dev.host_staging_buffer[i] = static_cast<std::byte>(i % 256);
			}

			dev_id++;
		}
		ensure(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &prior_mask) == 0);

		// transfer data from host to device, individually, contiguously
		for(int i = 0; i < total_runs; i++) {
			for(auto& dev : g_devices) {
				{
					dev.q.wait();
					const auto start = std::chrono::high_resolution_clock::now();
					auto ev = dev.q.memcpy(dev.dev_buffer, dev.host_buffer, total_bytes);
					ev.wait();
					const auto end = std::chrono::high_resolution_clock::now();
					if(i >= num_warmups) { dev.linear_h_to_d_time += end - start; }
				}

				{
					dev.q.wait();
					const auto start = std::chrono::high_resolution_clock::now();
					auto ev = dev.q.memcpy(dev.host_buffer, dev.dev_buffer, total_bytes);
					ev.wait();
					const auto end = std::chrono::high_resolution_clock::now();
					if(i >= num_warmups) { dev.linear_d_to_h_time += end - start; }
				}
			}
		}

		// timing data for each pair of devices
		std::vector<std::vector<std::pair<std::chrono::duration<double>, std::chrono::duration<double>>>> device_pair_times;
		device_pair_times.resize(g_devices.size());
		for(auto& v : device_pair_times) {
			v.resize(g_devices.size());
		}

		// transfer data from device to device, individually, contiguously
		for(int i = 0; i < total_runs; i++) {
			for(int source_idx = 0; source_idx < g_devices.size(); ++source_idx) {
				for(int target_idx = 0; target_idx < g_devices.size(); ++target_idx) {
					auto& source = g_devices[source_idx];
					auto& target = g_devices[target_idx];
					if(source_idx == target_idx) { continue; }

					{
						source.q.wait();
						const auto start = std::chrono::high_resolution_clock::now();
						auto ev = source.q.memcpy(target.dev_buffer, source.dev_buffer, total_bytes);
						ev.wait();
						const auto end = std::chrono::high_resolution_clock::now();
						if(i >= num_warmups) { device_pair_times[source_idx][target_idx].first += end - start; }
					}

					{
						source.q.wait();
						const auto start = std::chrono::high_resolution_clock::now();
						auto ev = source.q.memcpy(source.dev_buffer, target.dev_buffer, total_bytes);
						ev.wait();
						const auto end = std::chrono::high_resolution_clock::now();
						if(i >= num_warmups) { device_pair_times[source_idx][target_idx].second += end - start; }
					}
				}
			}
		}


		// output results
		{
			int i = 0;
			for(const auto& dev : g_devices) {
				std::cout << std::format(
				    "D{} H2D: {:10.2f} GB/s\n", i, (static_cast<double>(total_bytes) * num_repeats) / dev.linear_h_to_d_time.count() / 1e9);
				std::cout << std::format(
				    "D{} D2H: {:10.2f} GB/s\n", i, (static_cast<double>(total_bytes) * num_repeats) / dev.linear_d_to_h_time.count() / 1e9);
				i++;
			}

			// output device to device time matrix
			std::cout << "D2D matrix source -> dest:\n";
			for(int source_idx = 0; source_idx < g_devices.size(); ++source_idx) {
				for(int target_idx = 0; target_idx < g_devices.size(); ++target_idx) {
					std::cout << std::format(
					    "{:5.2f}, ", (static_cast<double>(total_bytes) * num_repeats) / device_pair_times[source_idx][target_idx].first.count() / 1e9);
				}
				std::cout << std::endl;
			}
			std::cout << "D2D matrix dest -> source:\n";
			for(int source_idx = 0; source_idx < g_devices.size(); ++source_idx) {
				for(int target_idx = 0; target_idx < g_devices.size(); ++target_idx) {
					std::cout << std::format(
					    "{:5.2f}, ", (static_cast<double>(total_bytes) * num_repeats) / device_pair_times[source_idx][target_idx].second.count() / 1e9);
				}
				std::cout << std::endl;
			};
		}

		return 0;
	}
