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
			COPYLIB_ENSURE(cpu_ids_split.size() == total_gpu_count, "Invalid number of CPU IDs provided in COPYLIB_ALLOC_CPU_IDS: {} (expected {})",
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
	ret += std::format("  2D copy available: {}\n", is_2d_copy_available());
	ret += std::format(" D2D copy available: {}\n", is_device_to_device_copy_available());
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
		dev.queue.wait_and_throw();
	}
}

executor::executor(int64_t buffer_size) : buffer_size(buffer_size) {
#ifdef SIMSYCL_VERSION
	auto sys_cfg = simsycl::get_default_system_config();
	sys_cfg.devices.emplace("gpu2", sys_cfg.devices.cbegin()->second);
	sys_cfg.devices.emplace("gpu3", sys_cfg.devices.cbegin()->second);
	sys_cfg.devices.emplace("gpu4", sys_cfg.devices.cbegin()->second);
	simsycl::configure_system(sys_cfg);
#endif

	const int64_t total_bytes = buffer_size;

	gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
	cpu_set_t prior_mask;
	CPU_ZERO(&prior_mask);
	COPYLIB_ENSURE(pthread_getaffinity_np(pthread_self(), sizeof(prior_mask), &prior_mask) == 0, "Failed to get CPU affinity");

	// allocate queues and device buffers
	devices.reserve(gpu_devices.size());
	int dev_id = 0;
	for(const auto& device : gpu_devices) {
		auto& dev = devices.emplace_back(sycl::queue(device));
		dev.dev_buffer = sycl::malloc_device<std::byte>(total_bytes, dev.queue);
		dev.staging_buffer = sycl::malloc_device<std::byte>(total_bytes, dev.queue);
		COPYLIB_ENSURE(dev.dev_buffer != nullptr, "Failed to allocate device buffer");
		COPYLIB_ENSURE(dev.staging_buffer != nullptr, "Failed to allocate device staging buffer");

		cpu_set_t mask_for_device;
		CPU_ZERO(&mask_for_device);
		const auto cpu_id = get_cpu_for_gpu_alloc(dev_id, gpu_devices.size());
		CPU_SET(cpu_id, &mask_for_device);
		COPYLIB_ENSURE(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask_for_device) == 0, "Failed to set CPU affinity");

		dev.host_buffer = sycl::malloc_host<std::byte>(total_bytes, dev.queue);
		dev.host_staging_buffer = sycl::malloc_host<std::byte>(total_bytes, dev.queue);
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
}

device::~device() {
	queue.wait_and_throw();

	sycl::free(dev_buffer, queue);
	sycl::free(staging_buffer, queue);
	sycl::free(host_buffer, queue);
	sycl::free(host_staging_buffer, queue);
}

sycl::queue& executor::get_queue(device_id id) {
	COPYLIB_ENSURE(static_cast<int>(id) >= 0 && static_cast<size_t>(id) < devices.size(), "Invalid device id: {} ({} device(s) available)", id, devices.size());
	return devices[static_cast<int>(id)].queue;
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

template <typename T, typename IdxType>
void copy_with_kernel_impl(sycl::queue& q, const copy_spec& spec, IdxType preferred_wg_size) {
	const T* src = reinterpret_cast<T*>(spec.source_layout.base_ptr() + spec.source_layout.offset);
	T* tgt = reinterpret_cast<T*>(spec.target_layout.base_ptr() + spec.target_layout.offset);

	const IdxType extent = spec.source_layout.total_bytes() / sizeof(T);
	IdxType wg_size = preferred_wg_size;
	while(extent % wg_size != 0) {
		wg_size /= 2;
	}
	const sycl::nd_range<1> ndr{extent, wg_size};
	if(spec.source_layout.fragment_count == spec.target_layout.fragment_count) {
		const IdxType frag_elems = spec.source_layout.fragment_length / sizeof(T);
		const IdxType src_stride = spec.source_layout.effective_stride() / sizeof(T);
		const IdxType tgt_stride = spec.target_layout.effective_stride() / sizeof(T);
		// sadly, all this sillyness is actually measurably faster
		if(frag_elems == 1) {
			if(tgt_stride == 1) {
				q.parallel_for(ndr, [=](sycl::nd_item<1> idx) { //
					const IdxType i = idx.get_global_id(0);
					const IdxType src_i = i * src_stride;
					tgt[i] = src[src_i];
				});
			} else if(src_stride == 1) {
				q.parallel_for(ndr, [=](sycl::nd_item<1> idx) { //
					const IdxType i = idx.get_global_id(0);
					const IdxType tgt_i = i * tgt_stride;
					tgt[tgt_i] = src[i];
				});
			} else {
				q.parallel_for(ndr, [=](sycl::nd_item<1> idx) { //
					const IdxType i = idx.get_global_id(0);
					const IdxType src_i = i * src_stride;
					const IdxType tgt_i = i * tgt_stride;
					tgt[tgt_i] = src[src_i];
				});
			}
		} else {
			q.parallel_for(ndr, [=](sycl::nd_item<1> idx) {
				const IdxType i = idx.get_global_id(0);
				const IdxType frag = i / frag_elems;
				const IdxType id_in_frag = i % frag_elems;
				tgt[frag * tgt_stride + id_in_frag] = src[frag * src_stride + id_in_frag];
			});
		}
	} else {
		const IdxType src_frag_elems = spec.source_layout.fragment_length / sizeof(T);
		const IdxType tgt_frag_elems = spec.target_layout.fragment_length / sizeof(T);
		const IdxType src_stride = spec.source_layout.effective_stride() / sizeof(T);
		const IdxType tgt_stride = spec.target_layout.effective_stride() / sizeof(T);
		q.parallel_for(ndr, [=](sycl::nd_item<1> idx) {
			const IdxType i = idx.get_global_id(0);
			const IdxType src_frag = i / src_frag_elems;
			const IdxType tgt_frag = i / tgt_frag_elems;
			tgt[tgt_frag * tgt_stride + i % tgt_frag_elems] = src[src_frag * src_stride + i % src_frag_elems];
		});
	}
}

template <typename T>
void copy_with_kernel_impl(sycl::queue& q, const copy_spec& spec, int32_t preferred_wg_size) {
	int64_t max = std::numeric_limits<int32_t>::max();
	if(spec.source_layout.fragment_count < max && spec.source_layout.fragment_count < max && spec.source_layout.effective_stride() < max
	    && spec.target_layout.effective_stride() < max && spec.source_layout.fragment_length < max && spec.target_layout.fragment_length < max) {
		copy_with_kernel_impl<T, int32_t>(q, spec, preferred_wg_size);
	} else {
		copy_with_kernel_impl<T, int64_t>(q, spec, preferred_wg_size);
	}
}

void copy_with_kernel(sycl::queue& q, const copy_spec& spec, int32_t preferred_wg_size) {
	// case distinction based on fragment size
	const auto smaller_fragment_size = std::min(spec.source_layout.fragment_length, spec.target_layout.fragment_length);
	const auto smaller_stride = std::min(spec.source_layout.effective_stride(), spec.target_layout.effective_stride());
	if(smaller_fragment_size % sizeof(sycl::int16) == 0 && smaller_stride % sizeof(sycl::int16) == 0) {
		copy_with_kernel_impl<sycl::int16>(q, spec, preferred_wg_size);
	} else if(smaller_fragment_size % sizeof(sycl::int8) == 0 && smaller_stride % sizeof(sycl::int8) == 0) {
		copy_with_kernel_impl<sycl::int8>(q, spec, preferred_wg_size);
	} else if(smaller_fragment_size % sizeof(sycl::int4) == 0 && smaller_stride % sizeof(sycl::int4) == 0) {
		copy_with_kernel_impl<sycl::int4>(q, spec, preferred_wg_size);
	} else if(smaller_fragment_size % sizeof(sycl::int2) == 0 && smaller_stride % sizeof(sycl::int2) == 0) {
		copy_with_kernel_impl<sycl::int2>(q, spec, preferred_wg_size);
	} else if(smaller_fragment_size % sizeof(int32_t) == 0 && smaller_stride % sizeof(int32_t) == 0) {
		copy_with_kernel_impl<int32_t>(q, spec, preferred_wg_size);
	} else if(smaller_fragment_size % sizeof(int16_t) == 0 && smaller_stride % sizeof(int16_t) == 0) {
		copy_with_kernel_impl<int16_t>(q, spec, preferred_wg_size);
	} else {
		copy_with_kernel_impl<int8_t>(q, spec, preferred_wg_size);
	}
}

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

device_id execute_copy(executor& exec, const copy_spec& spec) {
	// for host <-> host copies, use memcpy
	if(spec.source_device == device_id::host && spec.target_device == device_id::host) {
		copy_via_repeated_1D_copies(
		    [](const std::byte* src, std::byte* tgt, int64_t length) { std::memcpy(tgt, src, length); }, spec.source_layout, spec.target_layout);
		return device_id::host;
	}
	const device_id device_to_use = spec.source_device == device_id::host ? spec.target_device : spec.source_device;
	auto& queue = exec.get_queue(device_to_use);
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
	return device_to_use;
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
				    .size = layout.total_bytes(),
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
				COPYLIB_ENSURE(staging_buffers[staging_idx].size == layout.total_bytes(), "Staging buffer size mismatch");
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

void execute_copy(executor& exec, const copy_plan& plan) {
	staging_fulfiller fulfiller(exec);
	for(auto spec : plan) {
		fulfiller.fulfill(spec);
		auto dev = execute_copy(exec, spec);
		if(dev != device_id::host) exec.get_queue(dev).wait_and_throw();
	}
}

void execute_copy(executor& exec, const parallel_copy_set& set) {
	// TODO: smarter staging reuse
	// TODO: actual parallelization
	staging_fulfiller fulfiller(exec);
	for(const auto& plan : set) {
		for(auto spec : plan) {
			fulfiller.fulfill(spec);
			auto dev = execute_copy(exec, spec);
			if(dev != device_id::host) exec.get_queue(dev).wait_and_throw();
		}
	}
}

} // namespace copylib