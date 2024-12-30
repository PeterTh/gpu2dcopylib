#include "copylib.hpp"    // IWYU pragma: keep
#include "test_utils.hpp" // IWYU pragma: keep

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace copylib;

void fill_uniform(executor& exec, device_id did, intptr_t buffer, size_t bytes, uint8_t value) {
	auto queue = exec.get_queue(did);
	auto ptr = reinterpret_cast<uint8_t*>(buffer);
	queue.fill(ptr, value, bytes).wait_and_throw();
}

void fill_source(executor& exec, device_id did, intptr_t buffer, int64_t offset, size_t bytes, uint32_t start_value) {
	COPYLIB_ENSURE(bytes % sizeof(uint32_t) == 0, "Invalid buffer size");
	COPYLIB_ENSURE(offset % sizeof(uint32_t) == 0, "Invalid offset");
	const auto count = bytes / sizeof(uint32_t);
	const auto ptr = reinterpret_cast<uint32_t*>(buffer + offset);
	auto queue = exec.get_queue(did);
	queue.submit([&](sycl::handler& cgh) { cgh.parallel_for(sycl::range<1>{count}, [=](sycl::id<1> idx) { ptr[idx[0]] = start_value + idx[0]; }); });
	queue.wait_and_throw();
}

bool validate_contents(executor& exec, device_id did, intptr_t buffer, int64_t offset, size_t bytes, uint32_t start_value) {
	COPYLIB_ENSURE(bytes % sizeof(uint32_t) == 0, "Invalid buffer size");
	const auto count = bytes / sizeof(uint32_t);
	auto ptr = reinterpret_cast<uint32_t*>(buffer);
	auto queue = exec.get_queue(did);
	auto* valid_ptr = sycl::malloc_shared<uint32_t>(1, queue);
	*valid_ptr = true;
	queue.parallel_for(sycl::range<1>{count}, [=](sycl::id<1> idx) {
		const uint32_t expected = start_value + static_cast<uint32_t>(idx[0]);
		const uint32_t valid = ptr[idx[0]] == expected;
		sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*valid_ptr).fetch_and(valid);
		if(!valid) { COPYLIB_KERNEL_DEBUG_PRINTF("Mismatch at index %ld: expected %u, got %u\n", idx[0], expected, ptr[idx[0]]); }
	});
	queue.wait_and_throw();
	bool valid = *valid_ptr;
	sycl::free(valid_ptr, queue);
	return valid;
}

TEST_CASE("basic copies can be executed", "[executor]") {
	const int64_t buffer_size = GENERATE(1024, 76);
	CAPTURE(buffer_size);
	executor exec(buffer_size * 2);

	const auto src_buffer = reinterpret_cast<intptr_t>(exec.get_buffer(device_id::d0));
	const data_layout source_layout{src_buffer, 0, buffer_size};
	fill_source(exec, device_id::d0, src_buffer, 0, buffer_size, 42);

	const auto tgt_buffer = src_buffer + buffer_size;
	const data_layout target_layout{tgt_buffer, 0, buffer_size};
	const copy_properties props = is_2d_copy_available() ? GENERATE(copy_properties::none, copy_properties::use_kernel, copy_properties::use_2D_copy)
	                                                     : GENERATE(copy_properties::none, copy_properties::use_kernel);
	CAPTURE(props);
	const copy_spec spec{device_id::d0, source_layout, device_id::d0, target_layout, props};
	execute_copy(exec, normalize(spec));
	exec.get_queue(device_id::d0).wait_and_throw();

	CHECK(validate_contents(exec, device_id::d0, tgt_buffer, 0, buffer_size, 42));
}

TEST_CASE("2D copies can be executed", "[executor]") {
	const int64_t buffer_size = 1024 * 128;
	executor exec(buffer_size * 2);

	auto src_buffer = reinterpret_cast<intptr_t>(exec.get_buffer(device_id::d0));
	const int64_t source_offset = GENERATE(0, 32);
	CAPTURE(source_offset);
	const int64_t source_frag_length = GENERATE(8, 32);
	CAPTURE(source_frag_length);
	const int64_t source_frag_count = GENERATE(16, 32);
	CAPTURE(source_frag_count);
	const int64_t source_stride = GENERATE(256, 512);
	CAPTURE(source_stride);
	const data_layout source_layout{src_buffer, source_offset, source_frag_length, source_frag_count, source_stride};

	fill_uniform(exec, device_id::d0, src_buffer, buffer_size, 77);
	for(int i = 0; i < source_frag_count; i++) {
		fill_source(exec, device_id::d0, src_buffer, source_layout.fragment_offset(i), source_layout.fragment_length, 42 + i * 100);
	}

	constexpr bool debug_print = false;
	if(debug_print) {
		// print source buffer contents for debugging, as uint32_t
		auto ptr = reinterpret_cast<uint32_t*>(src_buffer);
		for(size_t i = 0; i < source_layout.end_offset() / sizeof(uint32_t); i++) {
			printf("src %lu %p %u\n", i, ptr + i, ptr[i]);
		}
	}

	auto tgt_buffer = src_buffer + buffer_size;
	const double target_frag_factor = GENERATE(0.5, 1.0, 2.0);
	CAPTURE(target_frag_factor);
	const int64_t target_offset = GENERATE(0, 80);
	CAPTURE(target_offset);
	const int64_t target_frag_length = static_cast<int64_t>(source_frag_length * target_frag_factor);
	const int64_t target_frag_count = static_cast<int64_t>(source_frag_count / target_frag_factor);
	const int64_t target_stride = 384;
	const data_layout target_layout{tgt_buffer, target_offset, target_frag_length, target_frag_count, target_stride};

	fill_uniform(exec, device_id::d0, tgt_buffer, buffer_size, 66);

	const copy_properties props = (is_2d_copy_available() && target_frag_factor == 1.0)
	                                  ? GENERATE(copy_properties::none, copy_properties::use_kernel, copy_properties::use_2D_copy)
	                                  : GENERATE(copy_properties::none, copy_properties::use_kernel);
	CAPTURE(props);

	const copy_spec spec{device_id::d0, source_layout, device_id::d0, target_layout, props};
	REQUIRE(is_valid(spec));
	execute_copy(exec, normalize(spec));
	exec.get_queue(device_id::d0).wait_and_throw();

	if(debug_print) {
		auto ptr = reinterpret_cast<uint32_t*>(tgt_buffer);
		for(size_t i = 0; i < target_layout.end_offset() / sizeof(uint32_t); i++) {
			printf("tgt %lu %p %u\n", i, ptr + i, ptr[i]);
		}
	}

	{
		auto ptr = reinterpret_cast<uint32_t*>(tgt_buffer + target_offset);
		auto queue = exec.get_queue(device_id::d0);
		auto* valid_ptr = sycl::malloc_shared<uint32_t>(1, queue);
		*valid_ptr = true;
		const auto elem_count = target_layout.total_extent() / sizeof(uint32_t);
		queue.parallel_for(sycl::range<1>{elem_count}, [=](sycl::id<1> idx) {
			// check if this element is within the target layout
			const auto id = static_cast<int64_t>(idx[0]);
			const auto frag_idx = (id * static_cast<int64_t>(sizeof(uint32_t))) / target_stride;
			const auto frag_offset = (id * static_cast<int64_t>(sizeof(uint32_t))) % target_stride;
			if(frag_offset < target_frag_length && frag_idx < target_frag_count) {
				const auto elem_idx_byte = frag_idx * target_frag_length + frag_offset;
				const auto source_frag_idx = elem_idx_byte / source_frag_length;
				const auto source_frag_offset = elem_idx_byte % source_frag_length;
				const uint32_t expected = 42 + source_frag_idx * 100 + source_frag_offset / sizeof(uint32_t);
				const uint32_t valid = ptr[id] == expected;
				sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*valid_ptr).fetch_and(valid);
				if(!valid) {
					COPYLIB_KERNEL_DEBUG_PRINTF("Mismatch at index %ld (byte#%3ld): expected %u, got %u\n", id, elem_idx_byte, expected, ptr[idx[0]]);
				}
			} else {
				// if it is not in the layout, it must be unchanged from the initial fill
				const uint32_t expected = (66 << 24 | 66 << 16 | 66 << 8 | 66);
				const uint32_t valid = ptr[id] == expected;
				sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*valid_ptr).fetch_and(valid);
				if(!valid) { COPYLIB_KERNEL_DEBUG_PRINTF("Mismatch at index %ld: expected %u, got %u\n", id, expected, ptr[idx[0]]); }
			}
		});
		queue.wait_and_throw();

		bool valid = *valid_ptr;
		sycl::free(valid_ptr, queue);
		CHECK(valid);
	}
}