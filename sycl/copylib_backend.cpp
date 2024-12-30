#include "copylib_backend.hpp"

#include "copylib_support.hpp" // IWYU pragma: keep - this is needed for formatting output, IWYU is dumb

#ifdef SIMSYCL_VERSION
#include <simsycl/system.hh>
#endif

namespace copylib {

bool is_2d_copy_available() {
#if SYCL_EXT_ONEAPI_MEMCPY2D > 0
	return true;
#else
	return false;
#endif // SYCL_EXT_ONEAPI_MEMCPY2D
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

	const auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
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
		CPU_SET(32 * dev_id, &mask_for_device); // TODO fix hardcoded NUMA <-> device mapping
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

template <typename T>
void copy_with_kernel_impl(sycl::queue& q, const copy_spec& spec) {
	const T* src = reinterpret_cast<T*>(spec.source_layout.base_ptr() + spec.source_layout.offset);
	T* tgt = reinterpret_cast<T*>(spec.target_layout.base_ptr() + spec.target_layout.offset);
	const int64_t src_frag_elems = spec.source_layout.fragment_length / sizeof(T);
	const int64_t tgt_frag_elems = spec.target_layout.fragment_length / sizeof(T);
	const auto src_stride = spec.source_layout.effective_stride() / sizeof(T);
	const auto tgt_stride = spec.target_layout.effective_stride() / sizeof(T);
	q.parallel_for(sycl::range<1>{spec.source_layout.total_bytes() / sizeof(T)}, [=](sycl::id<1> idx) {
		const int64_t src_frag = idx[0] / src_frag_elems;
		const int64_t tgt_frag = idx[0] / tgt_frag_elems;
		tgt[tgt_frag * tgt_stride + idx[0] % tgt_frag_elems] = src[src_frag * src_stride + idx[0] % src_frag_elems];
	});
}

void copy_with_kernel(sycl::queue& q, const copy_spec& spec) {
	// case distinction based on fragment size
	const auto smaller_fragment_size = std::min(spec.source_layout.fragment_length, spec.target_layout.fragment_length);
	const auto smaller_stride = std::min(spec.source_layout.effective_stride(), spec.target_layout.effective_stride());
	if(smaller_fragment_size % sizeof(sycl::int16) == 0 && smaller_stride % sizeof(sycl::int16) == 0) {
		copy_with_kernel_impl<sycl::int16>(q, spec);
	} else if(smaller_fragment_size % sizeof(sycl::int8) == 0 && smaller_stride % sizeof(sycl::int8) == 0) {
		copy_with_kernel_impl<sycl::int8>(q, spec);
	} else if(smaller_fragment_size % sizeof(sycl::int4) == 0 && smaller_stride % sizeof(sycl::int4) == 0) {
		copy_with_kernel_impl<sycl::int4>(q, spec);
	} else if(smaller_fragment_size % sizeof(sycl::int2) == 0 && smaller_stride % sizeof(sycl::int2) == 0) {
		copy_with_kernel_impl<sycl::int2>(q, spec);
	} else if(smaller_fragment_size % sizeof(int32_t) == 0 && smaller_stride % sizeof(int32_t) == 0) {
		copy_with_kernel_impl<int32_t>(q, spec);
	} else if(smaller_fragment_size % sizeof(int16_t) == 0 && smaller_stride % sizeof(int16_t) == 0) {
		copy_with_kernel_impl<int16_t>(q, spec);
	} else {
		copy_with_kernel_impl<int8_t>(q, spec);
	}
}

void execute_copy(executor& exec, const copy_spec& spec) {
	auto& queue = exec.get_queue(spec.source_device);
	if(spec.properties & copy_properties::use_kernel) {
		copy_with_kernel(queue, spec);
	} else if(spec.properties & copy_properties::use_2D_copy) {
#if SYCL_EXT_ONEAPI_MEMCPY2D > 0
		const auto dst_ptr = spec.target_layout.base_ptr() + spec.target_layout.offset;
		const auto src_ptr = spec.source_layout.base_ptr() + spec.source_layout.offset;
		const auto effective_dst_stride = spec.target_layout.effective_stride();
		const auto effective_src_stride = spec.source_layout.effective_stride();
		const auto width = spec.source_layout.fragment_length;
		const auto count = spec.source_layout.fragment_count;
		queue.ext_oneapi_memcpy2d(dst_ptr, effective_dst_stride, src_ptr, effective_src_stride, width, count);
#else
		COPYLIB_ERROR("2D copy requested, but not supported by the backend");
#endif // SYCL_EXT_ONEAPI_MEMCPY2D
	} else {
		// repeated 1D copies
		const auto larger_fragment_count = std::max(spec.source_layout.fragment_count, spec.target_layout.fragment_count);
		const auto smaller_fragment_size = std::min(spec.source_layout.fragment_length, spec.target_layout.fragment_length);
		for(int64_t frag = 0; frag < larger_fragment_count; ++frag) {
			const auto src_factor = spec.source_layout.fragment_length / smaller_fragment_size;
			const auto tgt_factor = spec.target_layout.fragment_length / smaller_fragment_size;
			const auto src_fragment_id = frag / src_factor;
			const auto tgt_fragment_id = frag / tgt_factor;
			const auto src_offset_in_fragment = frag % src_factor * spec.target_layout.fragment_length;
			const auto src = spec.source_layout.base_ptr() + spec.source_layout.fragment_offset(src_fragment_id) + src_offset_in_fragment;
			const auto tgt_offset_in_fragment = frag % tgt_factor * spec.source_layout.fragment_length;
			const auto tgt = spec.target_layout.base_ptr() + spec.target_layout.fragment_offset(tgt_fragment_id) + tgt_offset_in_fragment;
			queue.copy(src, tgt, smaller_fragment_size);
		}
	}
}

class staging_fulfiller {
  public:
	staging_fulfiller(executor& exec) : exec(exec) {}

	void fulfill(data_layout& layout, device_id did) {
		if(layout.is_unplaced_staging()) {
			const auto staging_idx = -layout.base - 1;
			if(staging_buffers[staging_idx].buffer == nullptr) {
				staging_buffers[staging_idx].size = layout.total_bytes();
				staging_buffers[staging_idx].buffer = exec.get_staging_buffer(did) + current_staging_offset;
				current_staging_offset += staging_buffers[staging_idx].size + (staging_buffers[staging_idx].size % staging_alignment);
				COPYLIB_ENSURE(current_staging_offset <= exec.get_buffer_size(), "Staging buffer overflow");
			} else {
				COPYLIB_ENSURE(staging_buffers[staging_idx].size == layout.total_bytes(), "Staging buffer size mismatch");
			}
			layout.base = reinterpret_cast<intptr_t>(staging_buffers[staging_idx].buffer);
		}
	}

	void fulfill(copy_spec& spec) {
		fulfill(spec.source_layout, spec.source_device);
		fulfill(spec.target_layout, spec.target_device);
	}

  private:
	executor& exec;
	int64_t current_staging_offset = 0;
	static constexpr int64_t staging_alignment = 128;

	struct staging_info {
		int64_t size = 0;
		std::byte* buffer = nullptr;
	};
	std::vector<staging_info> staging_buffers = std::vector<staging_info>(-data_layout::min_staging_id);
};

void execute_copy(executor& exec, const copy_plan& plan) {
	staging_fulfiller fulfiller(exec);
	for(auto spec : plan) {
		fulfiller.fulfill(spec);
		execute_copy(exec, spec);
		exec.get_queue(spec.source_device).wait_and_throw();
	}
}

} // namespace copylib