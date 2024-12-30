#include "copylib.hpp"    // IWYU pragma: keep
#include "test_utils.hpp" // IWYU pragma: keep

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace copylib;

class validation_flag {
	uint32_t* valid_ptr;
	sycl::queue& queue;

  public:
	validation_flag(sycl::queue& queue) : queue(queue) {
		valid_ptr = sycl::malloc_shared<uint32_t>(1, queue);
		*valid_ptr = true;
	}
	~validation_flag() { sycl::free(valid_ptr, queue); }

	class captured_flag {
		uint32_t* valid_ptr;

	  public:
		captured_flag(validation_flag& from) : valid_ptr(from.valid_ptr) {}
		void update(bool valid) const {
			sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*valid_ptr).fetch_and(static_cast<uint32_t>(valid));
		}
	};

	captured_flag capture() { return captured_flag(*this); }
	bool valid() const { return *valid_ptr; }
};

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
	validation_flag valid_flag(queue);
	queue.parallel_for(sycl::range<1>{count}, [=, flag = valid_flag.capture()](sycl::id<1> idx) {
		const uint32_t expected = start_value + static_cast<uint32_t>(idx[0]);
		const uint32_t valid = ptr[idx[0]] == expected;
		flag.update(valid);
		if(!valid) { COPYLIB_KERNEL_DEBUG_PRINTF("Mismatch at index %ld: expected %u, got %u\n", idx[0], expected, ptr[idx[0]]); }
	});
	queue.wait_and_throw();
	return valid_flag.valid();
}

bool validate_target(executor& exec, device_id did, intptr_t tgt_buffer, const data_layout& target_layout, const data_layout& source_layout) {
	auto ptr = reinterpret_cast<uint32_t*>(tgt_buffer + target_layout.offset);
	auto queue = exec.get_queue(did);
	validation_flag valid_flag(queue);
	const auto elem_count = target_layout.total_extent() / sizeof(uint32_t);
	queue.parallel_for(sycl::range<1>{elem_count}, [=, flag = valid_flag.capture()](sycl::id<1> idx) {
		// check if this element is within the target layout
		const auto id = static_cast<int64_t>(idx[0]);
		const auto frag_idx = (id * static_cast<int64_t>(sizeof(uint32_t))) / target_layout.stride;
		const auto frag_offset = (id * static_cast<int64_t>(sizeof(uint32_t))) % target_layout.stride;
		const auto elem_idx_byte = frag_idx * target_layout.fragment_length + frag_offset;
		const auto source_frag_idx = elem_idx_byte / source_layout.fragment_length;
		const auto source_frag_offset = elem_idx_byte % source_layout.fragment_length;
		// either it is part of the layout, then it has the expected set value, otherwise it should still be the fill value
		const uint32_t expected = (frag_offset < target_layout.fragment_length && frag_idx < target_layout.fragment_count)
		                              ? 42 + source_frag_idx * 100 + source_frag_offset / sizeof(uint32_t)
		                              : (66 << 24 | 66 << 16 | 66 << 8 | 66);
		const uint32_t valid = ptr[id] == expected;
		flag.update(valid);
		if(!valid) { COPYLIB_KERNEL_DEBUG_PRINTF("Mismatch at index %ld (byte#%3ld): expected %u, got %u\n", id, elem_idx_byte, expected, ptr[idx[0]]); }
	});
	queue.wait_and_throw();
	return valid_flag.valid();
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
	for(int i = 0; i < source_layout.fragment_count; i++) {
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

	const device_id tgt_device = GENERATE(device_id::d0, device_id::d1);

	auto tgt_buffer = [&] {
		switch(tgt_device) {
		case device_id::d0: return reinterpret_cast<intptr_t>(exec.get_buffer(device_id::d0) + buffer_size);
		case device_id::d1: return reinterpret_cast<intptr_t>(exec.get_buffer(device_id::d1));
		default: COPYLIB_ERROR("Invalid target device: {}", static_cast<int>(tgt_device));
		}
	}();
	const double target_frag_factor = GENERATE(0.5, 1.0, 2.0);
	CAPTURE(target_frag_factor);
	const int64_t target_offset = GENERATE(0, 80);
	CAPTURE(target_offset);
	const int64_t target_frag_length = static_cast<int64_t>(source_frag_length * target_frag_factor);
	const int64_t target_frag_count = static_cast<int64_t>(source_frag_count / target_frag_factor);
	const int64_t target_stride = 384;
	const data_layout target_layout{tgt_buffer, target_offset, target_frag_length, target_frag_count, target_stride};

	fill_uniform(exec, tgt_device, tgt_buffer, buffer_size, 66);

	// this is admittedly a bit weird
	const bool copy_2d_ok = is_2d_copy_available() && target_frag_factor == 1.0, copy_kernel_ok = tgt_device == device_id::d0;
	const copy_properties props = [&] {
		if(copy_2d_ok && copy_kernel_ok) {
			return GENERATE(copy_properties::none, copy_properties::use_kernel, copy_properties::use_2D_copy);
		} else if(copy_2d_ok) {
			return GENERATE(copy_properties::none, copy_properties::use_2D_copy);
		} else if(copy_kernel_ok) {
			return GENERATE(copy_properties::none, copy_properties::use_kernel);
		} else {
			return copy_properties::none;
		}
	}();
	CAPTURE(props);

	const copy_spec spec{tgt_device, source_layout, device_id::d0, target_layout, props};
	REQUIRE(is_valid(spec));
	execute_copy(exec, normalize(spec));
	exec.get_queue(tgt_device).wait_and_throw();

	if(debug_print) {
		auto ptr = reinterpret_cast<uint32_t*>(tgt_buffer);
		for(size_t i = 0; i < target_layout.end_offset() / sizeof(uint32_t); i++) {
			printf("tgt %lu %p %u\n", i, ptr + i, ptr[i]);
		}
	}

	CHECK(validate_target(exec, tgt_device, tgt_buffer, target_layout, source_layout));
}

TEST_CASE("copy plans can be executed", "[executor]") {
	const int64_t buffer_size = 1024 * 128;
	executor exec(buffer_size * 2);

	const auto src_buffer = reinterpret_cast<intptr_t>(exec.get_buffer(device_id::d0));
	const data_layout source_layout{src_buffer, 0, 16, 20, 256};
	const auto tgt_buffer = reinterpret_cast<intptr_t>(exec.get_buffer(device_id::d0) + buffer_size);
	const data_layout target_layout{tgt_buffer, 0, 16, 20, 256};
	const auto spec = copy_spec{device_id::d0, source_layout, device_id::d0, target_layout};

	const copy_properties props = is_2d_copy_available() ? GENERATE(copy_properties::none, copy_properties::use_kernel, copy_properties::use_2D_copy)
	                                                     : GENERATE(copy_properties::none, copy_properties::use_kernel);
	CAPTURE(props);
	const copy_strategy strat{copy_type::staged, props, 0};
	basic_staging_provider staging_provider;
	auto copy_plan = apply_staging(spec, strat, staging_provider);
	REQUIRE(copy_plan.size() == 3);

	fill_uniform(exec, device_id::d0, src_buffer, buffer_size, 77);
	for(int i = 0; i < source_layout.fragment_count; i++) {
		fill_source(exec, device_id::d0, src_buffer, source_layout.fragment_offset(i), source_layout.fragment_length, 42 + i * 100);
	}
	fill_uniform(exec, device_id::d0, tgt_buffer, buffer_size, 66);

	execute_copy(exec, copy_plan);

	CHECK(validate_target(exec, device_id::d0, tgt_buffer, target_layout, source_layout));
}
