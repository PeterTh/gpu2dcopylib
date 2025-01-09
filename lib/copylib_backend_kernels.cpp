#include <copylib.hpp> // IWYU pragma: keep

namespace copylib {

// Directly using CUDA threadIdx.x does NOT actually change performance
#define INDEX_X idx.get_global_id(0)

template <typename T, typename IdxType>
void copy_with_kernel_impl(sycl::queue& q, const copy_spec& spec, IdxType preferred_wg_size) {
	const T* src = reinterpret_cast<T*>(spec.source_layout.base_ptr() + spec.source_layout.offset);
	T* tgt = reinterpret_cast<T*>(spec.target_layout.base_ptr() + spec.target_layout.offset);

	const IdxType extent = spec.source_layout.total_bytes() / sizeof(T);
	IdxType wg_size = preferred_wg_size;
	while(extent % wg_size != 0) {
		wg_size /= 2;
	}
	const sycl::nd_range<1> ndr{static_cast<size_t>(extent), static_cast<size_t>(wg_size)};
	if(spec.source_layout.fragment_count == spec.target_layout.fragment_count) {
		const IdxType frag_elems = spec.source_layout.fragment_length / sizeof(T);
		const IdxType src_stride = spec.source_layout.effective_stride() / sizeof(T);
		const IdxType tgt_stride = spec.target_layout.effective_stride() / sizeof(T);
		// sadly, all this sillyness is actually measurably faster, and the cases are very common
		if(frag_elems == 1) {
			if(tgt_stride == 1) {
				q.parallel_for(ndr, [=]([[maybe_unused]] sycl::nd_item<1> idx) { //
					const IdxType i = INDEX_X;
					const IdxType src_i = i * src_stride;
					tgt[i] = src[src_i];
				});
			} else if(src_stride == 1) {
				q.parallel_for(ndr, [=]([[maybe_unused]] sycl::nd_item<1> idx) { //
					const IdxType i = INDEX_X;
					const IdxType tgt_i = i * tgt_stride;
					tgt[tgt_i] = src[i];
				});
			} else {
				q.parallel_for(ndr, [=]([[maybe_unused]] sycl::nd_item<1> idx) { //
					const IdxType i = INDEX_X;
					const IdxType src_i = i * src_stride;
					const IdxType tgt_i = i * tgt_stride;
					tgt[tgt_i] = src[src_i];
				});
			}
		} else {
			q.parallel_for(ndr, [=]([[maybe_unused]] sycl::nd_item<1> idx) {
				const IdxType i = INDEX_X;
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
		q.parallel_for(ndr, [=]([[maybe_unused]] sycl::nd_item<1> idx) {
			const IdxType i = INDEX_X;
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

} // namespace copylib
