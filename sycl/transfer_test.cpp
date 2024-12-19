
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <format>
#include <sycl/sycl.hpp>

#define ensure(expr)                                                                                                                                           \
	if(!(expr)) {                                                                                                                                              \
		std::cerr << "Error on line " << __LINE__ << ": " << #expr << std::endl;                                                                               \
		std::exit(1);                                                                                                                                          \
	}

struct Device {
	sycl::queue q;
	std::byte* dev_buffer = nullptr;
	std::byte* host_buffer = nullptr;
	std::chrono::duration<double> linear_h_to_d_time{0};
	std::chrono::duration<double> linear_d_to_h_time{0};
};

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

struct DataLayout {
	std::byte* base = nullptr;
	int64_t offset = 0;
	int64_t fragment_length = 0;
	int64_t fragment_count = 0;
	int64_t stride = 0;
	int64_t total_bytes() const { return fragment_count * fragment_length; }
	int64_t total_extent() const { return offset + fragment_count * stride; }
	bool unit_stride() const { return fragment_length == stride; }
};

struct CopySpec {
	DeviceID source_device;
	DataLayout source_layout;
	DeviceID destination_device;
	DataLayout destination_layout;
};

struct CopyPlan {
	std::vector<CopySpec> operations;
};

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

	// allocate queues and device buffers
	std::vector<Device> devices;
	for(const auto& device : gpu_devices) {
		auto& dev = devices.emplace_back(sycl::queue(device));
		dev.dev_buffer = sycl::malloc_device<std::byte>(total_bytes, dev.q);
		ensure(dev.dev_buffer != nullptr);
		dev.host_buffer = sycl::malloc_host<std::byte>(total_bytes, dev.q);
		ensure(dev.host_buffer != nullptr);
	}

	// initialize data on host
	for(auto& dev : devices) {
		for(int i = 0; i < total_bytes; i++) {
			dev.host_buffer[i] = static_cast<std::byte>(i % 256);
		}
	}

	// transfer data from host to device, individually, contiguously
	for(int i = 0; i < total_runs; i++) {
		for(auto& dev : devices) {
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
	device_pair_times.resize(devices.size());
	for(auto& v : device_pair_times) {
		v.resize(devices.size());
	}

	// transfer data from device to device, individually, contiguously
	for(int i = 0; i < total_runs; i++) {
		for(int source_idx = 0; source_idx < devices.size(); ++source_idx) {
			for(int target_idx = 0; target_idx < devices.size(); ++target_idx) {
				auto& source = devices[source_idx];
				auto& target = devices[target_idx];
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
		for(const auto& dev : devices) {
			std::cout << std::format("D{} H2D: {:10.2f} GB/s\n", i, (static_cast<double>(total_bytes) * num_repeats) / dev.linear_h_to_d_time.count() / 1e9);
			std::cout << std::format("D{} D2H: {:10.2f} GB/s\n", i, (static_cast<double>(total_bytes) * num_repeats) / dev.linear_d_to_h_time.count() / 1e9);
			i++;
		}

		// output device to device time matrix
		std::cout << "D2D matrix source -> dest:\n";
		for(int source_idx = 0; source_idx < devices.size(); ++source_idx) {
			for(int target_idx = 0; target_idx < devices.size(); ++target_idx) {
				std::cout << std::format(
				    "{:5.2f}, ", (static_cast<double>(total_bytes) * num_repeats) / device_pair_times[source_idx][target_idx].first.count() / 1e9);
			}
			std::cout << std::endl;
		}
		std::cout << "D2D matrix dest -> source:\n";
		for(int source_idx = 0; source_idx < devices.size(); ++source_idx) {
			for(int target_idx = 0; target_idx < devices.size(); ++target_idx) {
				std::cout << std::format(
				    "{:5.2f}, ", (static_cast<double>(total_bytes) * num_repeats) / device_pair_times[source_idx][target_idx].second.count() / 1e9);
			}
			std::cout << std::endl;
		};
	}

	return 0;
}
