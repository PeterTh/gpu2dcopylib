#include "copylib_backend.hpp"

#include "copylib_support.hpp" // IWYU pragma: keep - this is needed for formatting output, IWYU is dumb

#ifdef SIMSYCL_VERSION
#include <simsycl/system.hh>
#endif

#if defined(__ACPP_ENABLE_CUDA_TARGET__) && defined(COPYLIB_CUDA)
#define ACPP_WITH_CUDA true
#else
#define ACPP_WITH_CUDA false
#endif

#if ACPP_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <thread>

#include <vendor/bs_thread_pool/bs_thread_pool.hpp>

namespace copylib {

std::string executor::get_sycl_impl_name() const {
#if defined(SIMSYCL_VERSION)
	return "SimSYCL";
#elif defined(__ADAPTIVECPP__)
	std::string ret = "AdaptiveCPP";
#if defined(__ACPP_ENABLE_CUDA_TARGET__)
	ret += " (CUDA)";
#endif // __ACPP_ENABLE_CUDA_TARGET__
	return ret;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
	return "IntelSYCL";
#else
	return "Unknown SYCL implementation";
#endif
}

bool executor::is_2d_copy_available() const {
#if SYCL_EXT_ONEAPI_MEMCPY2D > 0
	return true;
#else
	return ACPP_WITH_CUDA;
#endif // SYCL_EXT_ONEAPI_MEMCPY2D
}

bool executor::is_device_to_device_copy_available() const {
#if defined(SIMSYCL_VERSION)
	return true;
#elif defined(__ADAPTIVECPP__)
#if defined(__ACPP_ENABLE_CUDA_TARGET__)
	return true; // CUDA emulates p2p transfers even if ont available
#else
	return false; // assumption for now
#endif
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
	if(gpu_devices.empty()) { return false; }
	return gpu_devices.front().get_info<sycl::info::device::vendor>().find("NVIDIA") != std::string::npos;
#else
	return false;
#endif
}

bool executor::is_peer_memory_access_available() const {
#if defined(SIMSYCL_VERSION)
	return true;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
	static bool available = [&] {
		for(size_t dev_idx_a = 0; dev_idx_a < devices.size(); dev_idx_a++) {
			for(size_t dev_idx_b = 0; dev_idx_b < devices.size(); dev_idx_b++) {
				if(dev_idx_a == dev_idx_b) { continue; }
				if(devices[dev_idx_a].dev.ext_oneapi_can_access_peer(devices[dev_idx_b].dev)) {
					devices[dev_idx_a].dev.ext_oneapi_enable_peer_access(devices[dev_idx_b].dev);
				} else {
					return false;
				}
			}
		}
		return true;
	}();
	return available;
#else
	return false;
#endif
}

int32_t executor::get_preferred_wg_size() const {
	thread_local int32_t wg_size = -1;
	if(wg_size == -1) {
		if(gpu_devices.empty()) { return 32; }
		if(gpu_devices.front().get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos) {
			wg_size = 128;
		} else {
			wg_size = 32;
		}
	}
	return wg_size;
}

int get_cpu_for_gpu_alloc(int gpu_idx, size_t total_gpu_count) {
	constexpr int max_gpu_idx = static_cast<int>(device_id::count);
	COPYLIB_ENSURE(gpu_idx < max_gpu_idx && gpu_idx >= 0, "Invalid gpu index: {} (needs to be >=0 and <{})", gpu_idx, max_gpu_idx);
	COPYLIB_ENSURE(total_gpu_count <= max_gpu_idx, "Invalid total gpu count: {} (needs to be <{})", total_gpu_count, max_gpu_idx);
	thread_local bool initialized = false;
	thread_local std::array<int, max_gpu_idx> cpu_for_gpu;
	if(!initialized) {
		auto env_var = std::getenv("COPYLIB_ALLOC_CPU_IDS");
		if(env_var) {
			const auto cpu_ids = std::string(env_var);
			const auto cpu_ids_split = utils::split(cpu_ids, ',');
			COPYLIB_ENSURE(cpu_ids_split.size() >= total_gpu_count, "Insufficient number of CPU IDs provided in COPYLIB_ALLOC_CPU_IDS: {} (expected {})",
			    cpu_ids_split.size(), total_gpu_count);
			for(size_t i = 0; i < total_gpu_count; i++) {
				cpu_for_gpu[i] = std::stoi(cpu_ids_split[i]);
			}
		} else { // guess
			const auto hw_concurrency = std::thread::hardware_concurrency();
			const auto cores = hw_concurrency / 2; // we just assume 2 threads per core
			for(size_t i = 0; i < total_gpu_count; i++) {
				cpu_for_gpu[i] = cores / total_gpu_count * i;
			}
		}
	}
	return cpu_for_gpu[gpu_idx];
}

std::string executor::get_info() const {
	auto ret = std::format("Copylib executor with {} device(s) and buffer size {} bytes\n", devices.size(), buffer_size);
	ret += std::format("SYCL implementation: {}\n", get_sycl_impl_name());
	ret += std::format("2D copy: {}    D2D copy: {}    Peer access: {}    Preferred wg size: {}\n", //
	    is_2d_copy_available(), is_device_to_device_copy_available(), is_peer_memory_access_available(), get_preferred_wg_size());
	ret += std::format("Using {} queues per device\n", get_queues_per_device());
	for(size_t i = 0; i < devices.size(); i++) {
		ret += std::format("    Device {:2}: {} [{}]", i, //
		    gpu_devices[i].get_info<sycl::info::device::name>(), gpu_devices[i].get_info<sycl::info::device::vendor>());
		ret += std::format(" (host alloc on core {})\n", get_cpu_for_gpu_alloc(i, devices.size()));
	}
	return ret;
}

executor::possibility executor::can_copy(const copy_spec& spec) const {
	const bool d2d = is_device_to_device_copy_available();
	const bool two_d = is_2d_copy_available();
	const auto d2d_copy = spec.source_device != spec.target_device && spec.source_device != device_id::host && spec.target_device != device_id::host;
	if(d2d_copy) {
		if(spec.properties & copy_properties::use_kernel) { return possibility::needs_d2d_copy; } // TODO need to be more specific here later
		if(!d2d) { return possibility::needs_d2d_copy; }
	}
	if(!two_d && (spec.properties & copy_properties::use_2D_copy)) { return possibility::needs_2d_copy; }
	return possibility::possible;
}

executor::possibility executor::can_copy(const parallel_copy_set& cset) const {
	for(const auto& plan : cset) {
		for(const auto& spec : plan) {
			const auto res = can_copy(spec);
			if(res != possibility::possible) { return res; }
		}
	}
	return possibility::possible;
}

void executor::barrier() {
	for(auto& dev : devices) {
		for(auto& q : dev.queues) {
			q.wait_and_throw();
		}
	}
}

executor::executor(int64_t buffer_size, int64_t devices_needed, int64_t queues_per_device) : buffer_size(buffer_size) {
	COPYLIB_ENSURE(devices_needed > 0, "Need at least one device");
	COPYLIB_ENSURE(queues_per_device > 0, "Need at least one queue per device");

#ifdef SIMSYCL_VERSION
	auto sys_cfg = simsycl::get_default_system_config();
	sys_cfg.devices.emplace("gpu2", sys_cfg.devices.cbegin()->second);
	sys_cfg.devices.emplace("gpu3", sys_cfg.devices.cbegin()->second);
	sys_cfg.devices.emplace("gpu4", sys_cfg.devices.cbegin()->second);
	simsycl::configure_system(sys_cfg);
#endif

	const int64_t total_bytes = buffer_size;

	gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
	if(gpu_devices.size() < static_cast<size_t>(devices_needed)) {
		COPYLIB_ERROR("Not enough GPU devices available: {} ({} needed)", gpu_devices.size(), devices_needed);
	} else if(gpu_devices.size() > static_cast<size_t>(devices_needed)) {
		gpu_devices.resize(devices_needed); // don't waste time initializing more devices than needed
	}
	cpu_set_t prior_mask;
	CPU_ZERO(&prior_mask);
	COPYLIB_ENSURE(pthread_getaffinity_np(pthread_self(), sizeof(prior_mask), &prior_mask) == 0, "Failed to get CPU affinity");

	// allocate queues and device buffers
	devices.reserve(gpu_devices.size());
	int dev_id = 0;
	for(const auto& device : gpu_devices) {
		const sycl::property_list queue_properties = {
		    sycl::property::queue::in_order{},
#ifdef ACPP_EXT_COARSE_GRAINED_EVENTS
		    // minor perf improvement on AdaptiveCPP
		    sycl::property::queue::AdaptiveCpp_coarse_grained_events{},
#endif // ACPP_EXT_COARSE_GRAINED_EVENTS
#ifdef SYCL_EXT_INTEL_QUEUE_IMMEDIATE_COMMAND_LIST
		    // ~ 10% perf improvement on Intel GPUs in strided chunk copy set peak performance
		    sycl::ext::intel::property::queue::immediate_command_list{},
#endif // SYCL_EXT_INTEL_QUEUE_IMMEDIATE_COMMAND_LIST
		};

		std::vector<sycl::queue> queues;
		for(int64_t i = 0; i < queues_per_device; i++) {
			queues.emplace_back(sycl::queue(device, queue_properties));
		}
		auto& dev = devices.emplace_back(device, queues);
		auto& q = dev.queues[0];

		dev.dev_buffer = sycl::malloc_device<std::byte>(total_bytes, q);
		dev.staging_buffer = sycl::malloc_device<std::byte>(total_bytes, q);
		COPYLIB_ENSURE(dev.dev_buffer != nullptr, "Failed to allocate device buffer");
		COPYLIB_ENSURE(dev.staging_buffer != nullptr, "Failed to allocate device staging buffer");

		cpu_set_t mask_for_device;
		CPU_ZERO(&mask_for_device);
		const auto cpu_id = get_cpu_for_gpu_alloc(dev_id, gpu_devices.size());
		CPU_SET(cpu_id, &mask_for_device);
		COPYLIB_ENSURE(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask_for_device) == 0, "Failed to set CPU affinity");

		dev.host_buffer = sycl::malloc_host<std::byte>(total_bytes, q);
		dev.host_staging_buffer = sycl::malloc_host<std::byte>(total_bytes, q);
		COPYLIB_ENSURE(dev.host_buffer != nullptr, "Failed to allocate host buffer");
		COPYLIB_ENSURE(dev.host_staging_buffer != nullptr, "Failed to allocate host staging buffer");
		// initialize data on host
		for(int i = 0; i < total_bytes; i++) {
			dev.host_buffer[i] = static_cast<std::byte>(i % 256);
			dev.host_staging_buffer[i] = static_cast<std::byte>(i % 256);
		}

		dev_id++;
	}
	COPYLIB_ENSURE(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &prior_mask) == 0, "Failed to reset CPU affinity");
} // namespace copylib

device::~device() {
	for(auto& q : queues) {
		q.wait_and_throw();
	}

	auto& q = queues[0];
	sycl::free(dev_buffer, q);
	sycl::free(staging_buffer, q);
	sycl::free(host_buffer, q);
	sycl::free(host_staging_buffer, q);
}

sycl::queue& executor::get_queue(device_id id, int64_t queue_idx) {
	COPYLIB_ENSURE(static_cast<int>(id) >= 0 && static_cast<size_t>(id) < devices.size(), "Invalid device id: {} ({} device(s) available)", id, devices.size());
	auto& queues = devices[static_cast<int>(id)].queues;
	COPYLIB_ENSURE(queue_idx >= 0 && static_cast<size_t>(queue_idx) < queues.size(), "Invalid queue idx: {} ({} queue(s) available)", queue_idx, queues.size());
	return queues[queue_idx];
}

std::byte* executor::get_buffer(device_id id) {
	COPYLIB_ENSURE(static_cast<int>(id) >= 0 && static_cast<size_t>(id) < devices.size(), "Invalid device id: {} ({} device(s) available)", id, devices.size());
	return devices[static_cast<int>(id)].dev_buffer;
}
std::byte* executor::get_staging_buffer(device_id id) {
	COPYLIB_ENSURE(static_cast<int>(id) >= 0 && static_cast<size_t>(id) < devices.size(), "Invalid device id: {} ({} device(s) available)", id, devices.size());
	return devices[static_cast<int>(id)].staging_buffer;
}

std::byte* executor::get_host_buffer(device_id id) {
	COPYLIB_ENSURE(static_cast<int>(id) >= 0 && static_cast<size_t>(id) < devices.size(), "Invalid device id: {} ({} device(s) available)", id, devices.size());
	return devices[static_cast<int>(id)].host_buffer;
}
std::byte* executor::get_host_staging_buffer(device_id id) {
	COPYLIB_ENSURE(static_cast<int>(id) >= 0 && static_cast<size_t>(id) < devices.size(), "Invalid device id: {} ({} device(s) available)", id, devices.size());
	return devices[static_cast<int>(id)].host_staging_buffer;
}

void copy_with_kernel(sycl::queue& q, const copy_spec& spec, int32_t preferred_wg_size);

template <typename CopyFun>
void copy_via_repeated_1D_copies(CopyFun fun, const data_layout& source_layout, const data_layout& target_layout) {
	const auto larger_fragment_count = std::max(source_layout.fragment_count, target_layout.fragment_count);
	const auto smaller_fragment_size = std::min(source_layout.fragment_length, target_layout.fragment_length);
	for(int64_t frag = 0; frag < larger_fragment_count; ++frag) {
		const auto src_factor = source_layout.fragment_length / smaller_fragment_size;
		const auto tgt_factor = target_layout.fragment_length / smaller_fragment_size;
		const auto src_fragment_id = frag / src_factor;
		const auto tgt_fragment_id = frag / tgt_factor;
		const auto src_offset_in_fragment = frag % src_factor * target_layout.fragment_length;
		const auto src = source_layout.base_ptr() + source_layout.fragment_offset(src_fragment_id) + src_offset_in_fragment;
		const auto tgt_offset_in_fragment = frag % tgt_factor * source_layout.fragment_length;
		const auto tgt = target_layout.base_ptr() + target_layout.fragment_offset(tgt_fragment_id) + tgt_offset_in_fragment;
		fun(src, tgt, smaller_fragment_size);
	}
}

executor::target execute_copy(executor& exec, const copy_spec& spec, int64_t queue_idx, const executor::target last_target) {
	constexpr bool debug = false;
	const device_id last_device = last_target.did;
	if(debug) utils::err_print("{}:\n  -> last_device is {}\n", spec, last_device);

	//  for host <-> host copies, use memcpy
	if(spec.source_device == device_id::host && spec.target_device == device_id::host) {
		if(debug) utils::err_print("  -> h2h\n");
		if(last_device != device_id::host && last_device != device_id::count) {
			if(debug) utils::err_print("  -> waiting on {}\n", last_device);
			exec.get_queue(last_device).wait_and_throw();
		}
		copy_via_repeated_1D_copies(
		    [](const std::byte* src, std::byte* tgt, int64_t length) { std::memcpy(tgt, src, length); }, spec.source_layout, spec.target_layout);
		return {device_id::host, 0};
	}

	const device_id device_to_use = spec.source_device == device_id::host ? spec.target_device : spec.source_device;
	const executor::target target{device_to_use, queue_idx};

	if(debug) utils::err_print("  -> performing copy on queue for device {}\n", device_to_use);
	if(last_target != target && last_device != device_id::count && last_device != device_id::host) {
		// utils::err_print("  -> waiting on {}\n", last_device);
		exec.get_queue(last_target).wait_and_throw();
	}

	auto& queue = exec.get_queue(target);

	// technically, one could use a kernel for copies involving the host on some hw/sw stacks, but we'll ignore that for now
	if(spec.properties & copy_properties::use_kernel && spec.source_device != device_id::host && spec.target_device != device_id::host) {
		copy_with_kernel(queue, spec, exec.get_preferred_wg_size());
	} else if(spec.properties & copy_properties::use_2D_copy) {
#if SYCL_EXT_ONEAPI_MEMCPY2D > 0
		const auto dst_ptr = spec.target_layout.base_ptr() + spec.target_layout.offset;
		const auto src_ptr = spec.source_layout.base_ptr() + spec.source_layout.offset;
		const auto effective_dst_stride = spec.target_layout.effective_stride();
		const auto effective_src_stride = spec.source_layout.effective_stride();
		const auto width = spec.source_layout.fragment_length;
		const auto count = spec.source_layout.fragment_count;
		queue.ext_oneapi_memcpy2d(dst_ptr, effective_dst_stride, src_ptr, effective_src_stride, width, count);
#elif ACPP_WITH_CUDA
		const cudaMemcpyKind kind = [&] {
			if(spec.source_device == device_id::host && spec.target_device != device_id::host) {
				return cudaMemcpyHostToDevice;
			} else if(spec.source_device != device_id::host && spec.target_device == device_id::host) {
				return cudaMemcpyDeviceToHost;
			} else {
				return cudaMemcpyDeviceToDevice;
			}
		}();
		queue.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle handle) {
			const auto& stream = handle.get_native_queue<sycl::backend::cuda>();
			cudaMemcpy2DAsync(spec.target_layout.base_ptr() + spec.target_layout.offset, spec.target_layout.effective_stride(), //
			    spec.source_layout.base_ptr() + spec.source_layout.offset, spec.source_layout.effective_stride(),               //
			    spec.source_layout.fragment_length, spec.source_layout.fragment_count, kind, stream);
		});
#else
		COPYLIB_ERROR("2D copy requested, but not supported by the backend");
#endif // SYCL_EXT_ONEAPI_MEMCPY2D
	} else {
		copy_via_repeated_1D_copies(
		    [&](const std::byte* src, std::byte* tgt, int64_t length) { queue.copy(src, tgt, length); }, spec.source_layout, spec.target_layout);
	}
	return target;
}

class staging_fulfiller {
  public:
	staging_fulfiller(executor& exec) : exec(exec) {}

	void fulfill(data_layout& layout) {
		if(layout.is_unplaced_staging()) {
			const auto staging_idx = layout.staging.index;
			auto staging_it = staging_buffers.find(staging_idx);
			if(staging_it == staging_buffers.end()) {
				const auto did = layout.staging.did;
				const bool host = layout.staging.on_host;
				COPYLIB_ENSURE(did != device_id::host, "Device id for staging cannot be host");
				staging_info info{
				    .size = layout.total_extent(),
				    .device = did,
				    .on_host = host,
				};
				if(host) {
					info.buffer = exec.get_host_staging_buffer(did) + current_host_staging_offsets[static_cast<size_t>(did)];
					current_host_staging_offsets[static_cast<size_t>(did)] += info.size + staging_alignment % info.size;
					COPYLIB_ENSURE(current_host_staging_offsets[static_cast<size_t>(did)] <= exec.get_buffer_size(),
					    "Staging buffer overflow on host for device {}", static_cast<int>(did));
				} else {
					info.buffer = exec.get_staging_buffer(did) + current_staging_offsets[static_cast<size_t>(did)];
					current_staging_offsets[static_cast<size_t>(did)] += info.size + staging_alignment % info.size;
					COPYLIB_ENSURE(current_staging_offsets[static_cast<size_t>(did)] <= exec.get_buffer_size(), "Staging buffer overflow for device {}",
					    static_cast<int>(did));
				}
				staging_it = staging_buffers.emplace(staging_idx, info).first;
			} else {
				COPYLIB_ENSURE(staging_buffers[staging_idx].size == layout.total_extent(), "Staging buffer size mismatch");
				COPYLIB_ENSURE(staging_buffers[staging_idx].device == layout.staging.did, "Staging buffer device mismatch");
				COPYLIB_ENSURE(staging_buffers[staging_idx].on_host == layout.staging.on_host, "Staging buffer host flag mismatch");
			}
			layout.base = reinterpret_cast<intptr_t>(staging_it->second.buffer);
		}
	}

	void fulfill(copy_spec& spec) {
		fulfill(spec.source_layout);
		fulfill(spec.target_layout);
	}

  private:
	executor& exec;
	std::vector<int64_t> current_staging_offsets = std::vector<int64_t>(static_cast<int>(device_id::count), 0);
	std::vector<int64_t> current_host_staging_offsets = std::vector<int64_t>(static_cast<int>(device_id::count), 0);
	static constexpr int64_t staging_alignment = 128;

	struct staging_info {
		int64_t size = 0;
		device_id device = device_id::d0;
		bool on_host = false;
		std::byte* buffer = nullptr;
	};
	std::unordered_map<decltype(staging_id::index), staging_info> staging_buffers;
};

class noop_fulfiller {
  public:
	void fulfill(copy_spec&) {}
};

template <typename T>
concept StagingFulfiller = requires(T f, copy_spec c) {
	{ f.fulfill(c) };
};

void execute_plan_impl(executor& exec, const copy_plan& plan, StagingFulfiller auto& fulfiller, int64_t queue_idx) {
	executor::target last_target = executor::null_target;
	for(auto spec : plan) {
		fulfiller.fulfill(spec);
		last_target = execute_copy(exec, spec, queue_idx, last_target);
	}
}

void execute_copy(executor& exec, const copy_plan& plan) {
	staging_fulfiller fulfiller(exec);
	execute_plan_impl(exec, plan, fulfiller, 0);
}

void execute_copy(executor& exec, const parallel_copy_set& set) {
	// TODO: smarter staging reuse
	// TODO make this better and more testable:
	//      - have a seperate type for executable copy sets, which are already staged and split into appropriate parts
	//      - test the splitting and staging logic separately
	const int64_t parts_count = exec.get_queues_per_device();
	static BS::thread_pool pool(parts_count);

	const int64_t total_plans = set.size();

	staging_fulfiller fulfiller(exec);
	std::vector<parallel_copy_set> fulfilled_sets(parts_count);
	std::vector<std::future<void>> futures;
	int64_t current_set_idx = 0;
	int64_t sets_added_to_current = 0;
	std::atomic<int64_t> plans_executed = 0;
	for(auto& plan : set) {
		copy_plan fulfilled_plan = plan;
		for(auto& spec : fulfilled_plan) {
			fulfiller.fulfill(spec);
		}
		fulfilled_sets[current_set_idx].insert(fulfilled_plan);
		sets_added_to_current++;

		const int64_t sets_to_add_to_current = total_plans / parts_count //
		                                       + ((current_set_idx < total_plans % parts_count) ? 1 : 0);
		if(sets_added_to_current >= sets_to_add_to_current) {
			futures.push_back(pool.submit_task([&, current_set_idx]() {
				noop_fulfiller ful;
				for(auto& plan : fulfilled_sets[current_set_idx]) {
					execute_plan_impl(exec, plan, ful, current_set_idx);
					plans_executed++;
				}
			}));
			current_set_idx++;
			sets_added_to_current = 0;
		}
	}
	for(auto& f : futures) {
		f.wait();
	}
	COPYLIB_ENSURE(plans_executed == total_plans, "Not all plans executed ({} of {})", plans_executed.load(), total_plans);
}

} // namespace copylib