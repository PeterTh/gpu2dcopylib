#include "copylib.hpp"

#include <chrono>
#include <cstdio>
#include <sched.h>

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
	ensure(pthread_getaffinity_np(pthread_self(), sizeof(prior_mask), &prior_mask) == 0, "Failed to get CPU affinity");

	// allocate queues and device buffers
	int dev_id = 0;
	for(const auto& device : gpu_devices) {
		auto& dev = g_devices.emplace_back(sycl::queue(device));
		dev.dev_buffer = sycl::malloc_device<std::byte>(total_bytes, dev.q);
		dev.staging_buffer = sycl::malloc_device<std::byte>(total_bytes, dev.q);
		ensure(dev.dev_buffer != nullptr, "Failed to allocate device buffer");
		ensure(dev.staging_buffer != nullptr, "Failed to allocate device staging buffer");

		cpu_set_t mask_for_device;
		CPU_ZERO(&mask_for_device);
		CPU_SET(32 * dev_id, &mask_for_device); // TODO fix hardcoded NUMA <-> device mapping
		ensure(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask_for_device) == 0, "Failed to set CPU affinity");

		dev.host_buffer = sycl::malloc_host<std::byte>(total_bytes, dev.q);
		dev.host_staging_buffer = sycl::malloc_host<std::byte>(total_bytes, dev.q);
		ensure(dev.host_buffer != nullptr, "Failed to allocate host buffer");
		ensure(dev.host_staging_buffer != nullptr, "Failed to allocate host staging buffer");
		// initialize data on host
		for(int i = 0; i < total_bytes; i++) {
			dev.host_buffer[i] = static_cast<std::byte>(i % 256);
			dev.host_staging_buffer[i] = static_cast<std::byte>(i % 256);
		}

		dev_id++;
	}
	ensure(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &prior_mask) == 0, "Failed to reset CPU affinity");

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
		for(size_t source_idx = 0; source_idx < g_devices.size(); ++source_idx) {
			for(size_t target_idx = 0; target_idx < g_devices.size(); ++target_idx) {
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
			std::cout << std::format("D{} H2D: {:10.2f} GB/s\n", i, (static_cast<double>(total_bytes) * num_repeats) / dev.linear_h_to_d_time.count() / 1e9);
			std::cout << std::format("D{} D2H: {:10.2f} GB/s\n", i, (static_cast<double>(total_bytes) * num_repeats) / dev.linear_d_to_h_time.count() / 1e9);
			i++;
		}

		// output device to device time matrix
		std::cout << "D2D matrix source -> dest:\n";
		for(size_t source_idx = 0; source_idx < g_devices.size(); ++source_idx) {
			for(size_t target_idx = 0; target_idx < g_devices.size(); ++target_idx) {
				std::cout << std::format(
				    "{:5.2f}, ", (static_cast<double>(total_bytes) * num_repeats) / device_pair_times[source_idx][target_idx].first.count() / 1e9);
			}
			std::cout << std::endl;
		}
		std::cout << "D2D matrix dest -> source:\n";
		for(size_t source_idx = 0; source_idx < g_devices.size(); ++source_idx) {
			for(size_t target_idx = 0; target_idx < g_devices.size(); ++target_idx) {
				std::cout << std::format(
				    "{:5.2f}, ", (static_cast<double>(total_bytes) * num_repeats) / device_pair_times[source_idx][target_idx].second.count() / 1e9);
			}
			std::cout << std::endl;
		};
	}

	return 0;
}
