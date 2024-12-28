#include "copylib_core.hpp"

#include "copylib.hpp" // IWYU pragma: keep
#include "utils.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_set>

namespace copylib {

bool is_valid(const data_layout& layout) { //
	return layout.fragment_length > 0 && layout.fragment_count > 0
	       && (layout.stride >= layout.fragment_length ||
	           // simple contiguous layout (allowed for 1D copies)
	           (layout.stride == 0 && layout.fragment_count == 1));
}

bool is_valid(const copy_spec& plan) {
	// check for overlapping source and target layouts
	if(plan.source_device == plan.target_device) {
		const auto source_end = plan.source_layout.offset + plan.source_layout.total_bytes();
		const auto target_end = plan.target_layout.offset + plan.target_layout.total_bytes();
		if(plan.source_layout.base == plan.target_layout.base) {
			if(plan.source_layout.offset < target_end && source_end > plan.target_layout.offset) { return false; }
		}
	}
	// we can't use both a kernel and a native 2D copy
	if(plan.properties & copy_properties::use_2D_copy && plan.properties & copy_properties::use_kernel) { return false; }
	// the layouts must be valid and compatible
	return is_valid(plan.source_layout) && is_valid(plan.target_layout) //
	       && plan.source_layout.total_bytes() == plan.target_layout.total_bytes();
}

bool is_valid(const copy_plan& plan) {
	// each indivudal copy must be valid
	if(!std::ranges::all_of(plan, [](const copy_spec& copy) { return is_valid(copy); })) { return false; }
	// the copies must connect properly
	return std::ranges::adjacent_find(plan, [](const copy_spec& a, const copy_spec& b) { return a.target_layout != b.source_layout; }) == plan.end();
}

bool is_valid(const parallel_copy_set& set) {
	// each individual copy plan must be valid
	return std::ranges::all_of(set, [](const copy_plan& plan) { return is_valid(plan); });
}

bool is_equivalent(const copy_plan& plan, const copy_spec& spec) {
	COPYLIB_ENSURE(is_valid(spec), "Invalid copy specification, cannot compare to plan: {}", spec);
	COPYLIB_ENSURE(is_valid(plan), "Invalid copy plan, cannot compare to spec: {}", plan);

	if(plan.empty()) { return false; }

	const auto& first_spec = plan.front();
	const auto& last_spec = plan.back();
	return first_spec.source_device == spec.source_device && first_spec.source_layout == spec.source_layout && last_spec.target_device == spec.target_device
	       && last_spec.target_layout == spec.target_layout;
}

bool is_equivalent(const parallel_copy_set& set, const copy_spec& spec) {
	COPYLIB_ENSURE(is_valid(spec), "Invalid copy specification, cannot compare to set: {}", spec);
	COPYLIB_ENSURE(is_valid(set), "Invalid copy set, cannot compare to spec: {}", set);

	int64_t source_start = std::numeric_limits<int64_t>::max();
	int64_t source_end = std::numeric_limits<int64_t>::min();
	int64_t source_copied = 0;
	int64_t target_start = std::numeric_limits<int64_t>::max();
	int64_t target_end = std::numeric_limits<int64_t>::min();
	int64_t target_copied = 0;

	const auto source_fragment_size = spec.source_layout.fragment_length;
	const auto source_stride = spec.source_layout.stride;
	const auto target_fragment_size = spec.target_layout.fragment_length;
	const auto target_stride = spec.target_layout.stride;

	for(const auto& plan : set) {
		COPYLIB_ENSURE(is_valid(plan), "Invalid copy plan in set, cannot compare to spec: {}", plan);
		const auto& first_spec = plan.front();
		const auto& last_spec = plan.back();

		if(first_spec.source_device != spec.source_device || first_spec.source_layout.base != spec.source_layout.base) { return false; }
		if(last_spec.target_device != spec.target_device || last_spec.target_layout.base != spec.target_layout.base) { return false; }
		if(!first_spec.source_layout.unit_stride()
		    && (first_spec.source_layout.fragment_length != source_fragment_size || first_spec.source_layout.stride != source_stride)) {
			return false;
		}
		if(!last_spec.target_layout.unit_stride()
		    && (last_spec.target_layout.fragment_length != target_fragment_size || last_spec.target_layout.stride != target_stride)) {
			return false;
		}

		source_start = std::min(source_start, first_spec.source_layout.offset);
		source_end = std::max(source_end, first_spec.source_layout.end_offset());
		source_copied += first_spec.source_layout.total_bytes();

		target_start = std::min(target_start, last_spec.target_layout.offset);
		target_end = std::max(target_end, last_spec.target_layout.end_offset());
		target_copied += last_spec.target_layout.total_bytes();
	}

	return source_start == spec.source_layout.offset && source_end == spec.source_layout.end_offset() && source_copied == spec.source_layout.total_bytes()
	       && target_start == spec.target_layout.offset && target_end == spec.target_layout.end_offset() && target_copied == spec.target_layout.total_bytes();
}

data_layout normalize(const data_layout& layout) {
	if(!layout.unit_stride() || layout.fragment_count == 1) { return layout; }
	const auto bytes = layout.total_bytes();
	return {layout.base, layout.offset, bytes, 1, bytes};
}

copy_spec normalize(const copy_spec& spec) {
	if(!spec.is_contiguous() || (spec.source_layout.fragment_count == 1 && spec.target_layout.fragment_count == 1)) { return spec; }
	return {spec.source_device, normalize(spec.source_layout), spec.target_device, normalize(spec.target_layout), spec.properties};
}

copy_spec apply_properties(const copy_spec& spec, const copy_properties& props) {
	copy_spec ret = spec;
	ret.properties = props;
	return ret;
}

// TODO: this function could probably be much less repetitive and smarter
parallel_copy_set apply_chunking(const copy_spec& spec, const copy_strategy& strategy) {
	COPYLIB_ENSURE(is_valid(spec), "Invalid copy specification, cannot chunk: {}", spec);
	if(strategy.chunk_size == 0) { return {{spec}}; }
	parallel_copy_set copy_set;

	// I) contiguous copies are relatively easy to chunk
	if(spec.source_layout.unit_stride() && spec.target_layout.unit_stride()) {
		const auto total_bytes = spec.source_layout.total_bytes();
		const auto num_chunks = (total_bytes + strategy.chunk_size - 1) / strategy.chunk_size;
		for(int64_t i = 0; i < num_chunks; i++) {
			const auto start_offset = i * strategy.chunk_size;
			const auto source_offset = spec.source_layout.offset + start_offset;
			const auto target_offset = spec.target_layout.offset + start_offset;
			const auto fragment_length = std::min(strategy.chunk_size, total_bytes - start_offset);
			copy_set.insert({{                                                                                     //
			    spec.source_device, {spec.source_layout.base, source_offset, fragment_length, 1, fragment_length}, //
			    spec.target_device, {spec.target_layout.base, target_offset, fragment_length, 1, fragment_length}}});
		}
		return copy_set;
	}

	// II) non-contiguous copy, split the fragments into chunks
	// it would be possible to act on a sub-fragment level, but currently the assumption is that
	// fragments occur for things like column copies of a 2D array, and therefore we expect the fragment size to be small compared to the chunk size

	// case 1: source is unit stride, target is non-unit stride
	if(spec.source_layout.unit_stride() && !spec.target_layout.unit_stride()) {
		COPYLIB_ENSURE(spec.target_layout.fragment_length <= strategy.chunk_size, "Cannot chunk, fragments too large for chunking ({} > {})",
		    spec.target_layout.fragment_length, strategy.chunk_size);
		const auto fragments_per_chunk = strategy.chunk_size / spec.target_layout.fragment_length;
		const auto num_chunks = spec.target_layout.fragment_count / fragments_per_chunk + //
		                        (spec.target_layout.fragment_count % fragments_per_chunk != 0 ? 1 : 0);
		const auto total_bytes_per_chunk = spec.target_layout.fragment_length * fragments_per_chunk;
		for(int64_t i = 0; i < num_chunks; i++) {
			const auto start_fragment = i * fragments_per_chunk;
			const auto end_fragment = std::min(start_fragment + fragments_per_chunk, spec.target_layout.fragment_count);
			const auto num_fragments = end_fragment - start_fragment;
			const auto source_offset = spec.source_layout.offset + start_fragment * spec.target_layout.fragment_length;
			const auto dest_offset = spec.target_layout.fragment_offset(start_fragment);
			copy_set.insert({{                                                                             //
			    spec.source_device, {spec.source_layout.base, source_offset, total_bytes_per_chunk, 1, 0}, //
			    spec.target_device, {spec.target_layout.base, dest_offset, spec.target_layout.fragment_length, num_fragments, spec.target_layout.stride}}});
		}
		return copy_set;
	}

	// case 2: source is non-unit stride, target is unit stride
	if(!spec.source_layout.unit_stride() && spec.target_layout.unit_stride()) {
		COPYLIB_ENSURE(spec.source_layout.fragment_length <= strategy.chunk_size, "Cannot chunk, fragments too large for chunking ({} > {})",
		    spec.source_layout.fragment_length, strategy.chunk_size);
		const auto fragments_per_chunk = strategy.chunk_size / spec.source_layout.fragment_length;
		const auto num_chunks = spec.source_layout.fragment_count / fragments_per_chunk + //
		                        (spec.source_layout.fragment_count % fragments_per_chunk != 0 ? 1 : 0);
		const auto total_bytes_per_chunk = spec.source_layout.fragment_length * fragments_per_chunk;
		for(int64_t i = 0; i < num_chunks; i++) {
			const auto start_fragment = i * fragments_per_chunk;
			const auto end_fragment = std::min(start_fragment + fragments_per_chunk, spec.source_layout.fragment_count);
			const auto num_fragments = end_fragment - start_fragment;
			const auto source_offset = spec.source_layout.fragment_offset(start_fragment);
			const auto dest_offset = spec.target_layout.offset + start_fragment * spec.source_layout.fragment_length;
			copy_set.insert({{                                                                                                                              //
			    spec.source_device, {spec.source_layout.base, source_offset, spec.source_layout.fragment_length, num_fragments, spec.source_layout.stride}, //
			    spec.target_device, {spec.target_layout.base, dest_offset, total_bytes_per_chunk, 1, 0}}});
		}
		return copy_set;
	}

	// case 3: both source and target are non-unit stride
	if(!spec.source_layout.unit_stride() && !spec.target_layout.unit_stride()) {
		const auto larger_fragment_length = std::max(spec.source_layout.fragment_length, spec.target_layout.fragment_length);
		const auto smaller_fragment_length = std::min(spec.source_layout.fragment_length, spec.target_layout.fragment_length);
		COPYLIB_ENSURE(larger_fragment_length <= strategy.chunk_size, "Cannot chunk, fragments too large for chunking ({} > {})", larger_fragment_length,
		    strategy.chunk_size);
		COPYLIB_ENSURE(larger_fragment_length % smaller_fragment_length == 0, "Cannot chunk, fragment sizes not compatible ({} % {} != 0)",
		    larger_fragment_length, smaller_fragment_length);
		const auto larger_fragments_per_chunk = strategy.chunk_size / larger_fragment_length;
		const auto smaller_fragments_per_larger_fragment = larger_fragment_length / smaller_fragment_length;
		const auto smaller_fragments_per_chunk = larger_fragments_per_chunk * smaller_fragments_per_larger_fragment;
		const auto count_of_larger_fragments = std::min(spec.source_layout.fragment_count, spec.target_layout.fragment_count);
		const auto num_chunks = count_of_larger_fragments / larger_fragments_per_chunk + //
		                        (count_of_larger_fragments % larger_fragments_per_chunk != 0 ? 1 : 0);
		for(int64_t i = 0; i < num_chunks; i++) {
			if(spec.source_layout.fragment_length > spec.target_layout.fragment_length) {
				const auto source_start_fragment = i * larger_fragments_per_chunk;
				COPYLIB_ENSURE(source_start_fragment < spec.source_layout.fragment_count, "Invalid source fragment index {} of {}", source_start_fragment,
				    spec.source_layout.fragment_count);
				const auto source_end_fragment = std::min(source_start_fragment + larger_fragments_per_chunk, spec.source_layout.fragment_count);
				const auto num_source_fragments = source_end_fragment - source_start_fragment;
				const auto source_offset = spec.source_layout.fragment_offset(source_start_fragment);
				const auto target_start_fragment = source_start_fragment * smaller_fragments_per_chunk;
				COPYLIB_ENSURE(target_start_fragment < spec.target_layout.fragment_count, "Invalid target fragment index {} of {}", target_start_fragment,
				    spec.target_layout.fragment_count);
				const auto target_end_fragment = source_end_fragment * smaller_fragments_per_chunk;
				const auto num_target_fragments = target_end_fragment - target_start_fragment;
				const auto target_offset = spec.target_layout.fragment_offset(target_start_fragment);
				copy_set.insert({{                                                                                                                         //
				    spec.source_device, {spec.source_layout.base, source_offset, larger_fragment_length, num_source_fragments, spec.source_layout.stride}, //
				    spec.target_device, {spec.target_layout.base, target_offset, smaller_fragment_length, num_target_fragments, spec.target_layout.stride}}});
			} else {
				const auto source_start_fragment = i * smaller_fragments_per_chunk;
				COPYLIB_ENSURE(source_start_fragment < spec.source_layout.fragment_count, "Invalid source fragment index {} of {}", source_start_fragment,
				    spec.source_layout.fragment_count);
				const auto source_end_fragment = std::min(source_start_fragment + smaller_fragments_per_chunk, spec.source_layout.fragment_count);
				const auto num_source_fragments = source_end_fragment - source_start_fragment;
				const auto source_offset = spec.source_layout.fragment_offset(source_start_fragment);
				const auto target_start_fragment = source_start_fragment / smaller_fragments_per_larger_fragment;
				COPYLIB_ENSURE(target_start_fragment < spec.target_layout.fragment_count, "Invalid target fragment index {} of {}", target_start_fragment,
				    spec.target_layout.fragment_count);
				const auto target_end_fragment = source_end_fragment / smaller_fragments_per_larger_fragment;
				const auto num_target_fragments = target_end_fragment - target_start_fragment;
				const auto target_offset = spec.target_layout.fragment_offset(target_start_fragment);
				copy_set.insert({{                                                                                                                          //
				    spec.source_device, {spec.source_layout.base, source_offset, smaller_fragment_length, num_source_fragments, spec.source_layout.stride}, //
				    spec.target_device, {spec.target_layout.base, target_offset, larger_fragment_length, num_target_fragments, spec.target_layout.stride}}});
			}
		}
		return copy_set;
	}
	COPYLIB_ERROR("Unexpected copy layout when chunking: {}", spec);
}

copy_plan apply_staging(const copy_spec& spec, const copy_strategy& strategy, const staging_buffer_provider& staging_provider) {
	COPYLIB_ENSURE(is_valid(spec), "Invalid copy specification, cannot stage: {}", spec);
	const auto proper_spec = apply_properties(spec, strategy.properties);
	if(strategy.type == copy_type::direct) { return {proper_spec}; }
	if(strategy.type != copy_type::staged) {
		COPYLIB_ERROR("Unknown copy strategy type: {}", strategy.type);
		return {proper_spec};
	}
	// if we are looking at a contiguous copy, we don't need to stage, but we need to normalize the layouts
	if(spec.is_contiguous()) { return {normalize(proper_spec)}; }

	// if the source is not unit stride, we need to stage the source
	std::optional<copy_spec> source_staging_copy;
	if(!spec.source_layout.unit_stride()) {
		const auto source_staging_buffer = staging_provider(spec.source_device, spec.source_layout.total_bytes());
		const data_layout staged_source_layout{source_staging_buffer, 0, spec.source_layout.total_bytes(), 1, spec.source_layout.total_bytes()};
		source_staging_copy.emplace(spec.source_device, spec.source_layout, spec.source_device, staged_source_layout, strategy.properties);
		COPYLIB_ENSURE(is_valid(source_staging_copy.value()), "Created invalid source staging copy {} from {}", source_staging_copy.value(), spec);
	}

	// if the target is not unit stride, we need to unstage the target
	std::optional<copy_spec> target_unstaging_copy;
	if(!spec.target_layout.unit_stride()) {
		const auto target_staging_buffer = staging_provider(spec.target_device, spec.target_layout.total_bytes());
		const data_layout staged_target_layout{target_staging_buffer, 0, spec.target_layout.total_bytes(), 1, spec.target_layout.total_bytes()};
		target_unstaging_copy.emplace(spec.target_device, staged_target_layout, spec.target_device, spec.target_layout, strategy.properties);
		COPYLIB_ENSURE(is_valid(target_unstaging_copy.value()), "Created invalid target unstaging copy {} from {}", target_unstaging_copy.value(), spec);
	}

	// now we can build the copy plan
	copy_plan plan;
	if(source_staging_copy.has_value() && target_unstaging_copy.has_value()) {
		const auto& src = source_staging_copy.value();
		const auto& tgt = target_unstaging_copy.value();
		plan.push_back(src);
		plan.emplace_back(src.source_device, src.target_layout, tgt.target_device, tgt.source_layout, strategy.properties);
		plan.push_back(tgt);
	} else if(source_staging_copy.has_value()) {
		const auto& src = source_staging_copy.value();
		plan.push_back(src);
		plan.emplace_back(src.target_device, src.target_layout, spec.target_device, spec.target_layout, strategy.properties);
	} else if(target_unstaging_copy.has_value()) {
		const auto& tgt = target_unstaging_copy.value();
		plan.emplace_back(spec.source_device, spec.source_layout, tgt.source_device, tgt.source_layout, strategy.properties);
		plan.push_back(tgt);
	} else {
		COPYLIB_ERROR("Something strange is afoot when staging: {}", spec);
	}
	return plan;
}

parallel_copy_set apply_staging(const parallel_copy_set& spec, const copy_strategy& strategy, const staging_buffer_provider& staging_provider) {
	parallel_copy_set copies;
	for(const auto& copy : spec) {
		COPYLIB_ENSURE(copy.size() == 1, "Cannot stage a copy set with plans consisting of more than one copy (plan: {})", copy);
		copies.insert(apply_staging(copy.front(), strategy, staging_provider));
	}
	return copies;
}

parallel_copy_set manifest_strategy(const copy_spec& spec, const copy_strategy& strategy, const staging_buffer_provider& staging_provider) {
	const auto chunked_copies = apply_chunking(spec, strategy);
	return apply_staging(chunked_copies, strategy, staging_provider);
}

} // namespace copylib
