#include "copylib.hpp" // IWYU pragma: keep

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace copylib;

namespace Catch {
template <>
struct StringMaker<copy_spec> {
	static std::string convert(const copy_spec& value) {
		return std::format("\n{}", value); // extra newline for comparison in test output
	}
};
} // namespace Catch

TEST_CASE("formatting of types", "[format]") {
	SECTION("device_id") {
		CHECK(std::format("{}", device_id::host) == "host");
		CHECK(std::format("{}", device_id::d0) == "d0");
		CHECK(std::format("{}", device_id::d5) == "d5");
	}
	SECTION("data_layout") {
		const data_layout layout{0, 0, 1024, 1, 1024};
		CHECK(std::format("{}", layout) == "{0x0+0, [1024 * 1, 1024]}");
		const data_layout staging_layout{-3, 0, 1024, 1};
		CHECK(std::format("{}", staging_layout) == "{S-3+0, [1024 * 1, 0]}");
	}
	SECTION("copy_properties") {
		CHECK(std::format("{}", copy_properties::none) == "");
		CHECK(std::format("{}", copy_properties::use_kernel) == "use_kernel");
		CHECK(std::format("{}", copy_properties::use_2D_copy) == "use_2D_copy");
		CHECK(std::format("{}", copy_properties::use_kernel | copy_properties::use_2D_copy) == "use_kernel,use_2D_copy");
	}
	SECTION("copy_spec") {
		const copy_spec spec{device_id::d0, {0, 42, 1024, 1, 1024}, device_id::d1, {0xdeadbeef, 0, 256, 4, 320}};
		CHECK(std::format("{}", spec) == "copy(d0{0x0+42, [1024 * 1, 1024]}, d1{0xdeadbeef+0, [256 * 4, 320]})");
	}
	SECTION("copy_type") {
		CHECK(std::format("{}", copy_type::direct) == "direct");
		CHECK(std::format("{}", copy_type::staged) == "staged");
	}
	SECTION("copy_strategy") {
		const copy_strategy strategy{copy_type::direct, copy_properties::use_kernel, 256};
		CHECK(std::format("{}", strategy) == "strategy(direct(use_kernel), chunk:256)");
	}
}

TEST_CASE("data layout validation", "[validation]") {
	CHECK(is_valid({0, 0, 1024, 1, 1024}));
	CHECK(is_valid({0, 0, 1024, 1, 0}));
	CHECK(!is_valid({0, 0, 1024, 2, 0}));
	CHECK(!is_valid({0, 0, 0, 0, 0}));
	CHECK(!is_valid({0, 0, 1024, 1, 512}));
}

TEST_CASE("copy spec validation", "[validation]") {
	const data_layout valid_layout{0, 0, 1024, 1, 1024};
	CHECK(is_valid({device_id::d0, valid_layout, device_id::d1, valid_layout}));
	CHECK(is_valid({device_id::d0, valid_layout, device_id::d0, {0, 1024, 1024, 1, 1024}}));
	CHECK(!is_valid({device_id::d0, valid_layout, device_id::d0, valid_layout}));          // overlapping source and target
	CHECK(!is_valid({device_id::d0, {0, 0, 0, 1, 1024}, device_id::d1, valid_layout}));    // invalid source layout
	CHECK(!is_valid({device_id::d0, valid_layout, device_id::d1, {0, 0, 1024, 1, 512}}));  // invalid target layout
	CHECK(!is_valid({device_id::d0, valid_layout, device_id::d1, {0, 0, 2048, 1, 2048}})); // different total bytes
	CHECK(!is_valid({device_id::d0, valid_layout, device_id::d1, {0, 0, 1024, 2, 1024}})); // different total bytes
	CHECK(is_valid({device_id::d0, valid_layout, device_id::d1, {0, 256, 512, 2, 512}}));  // fine!
	CHECK(!is_valid({device_id::d0, valid_layout, device_id::d1, valid_layout, copy_properties::use_2D_copy | copy_properties::use_kernel})); // can't have both
	CHECK(is_valid({device_id::d0, valid_layout, device_id::d1, valid_layout, copy_properties::use_2D_copy}));                                // fine!
}

TEST_CASE("copy plan validation", "[validation]") {
	const data_layout valid_layout{0, 0, 1024, 1, 1024};
	const copy_spec valid_spec{device_id::d0, valid_layout, device_id::d1, valid_layout};
	const copy_plan trivial_plan{valid_spec};
	CHECK(is_valid(trivial_plan));
	const copy_plan valid_plan{valid_spec, {device_id::d1, valid_layout, device_id::d2, valid_layout}};
	CHECK(is_valid(valid_plan));
	const copy_spec invalid_spec{device_id::d0, valid_layout, device_id::d1, {0, 0, 1024, 1, 512}};
	const copy_plan invalid_plan{valid_spec, invalid_spec};
	CHECK(!is_valid(invalid_plan));
	// check connectivity between steps
	const copy_plan invalid_plan2{valid_spec, {device_id::d2, valid_layout, device_id::d2, valid_layout}}; // wrong device
	CHECK(!is_valid(invalid_plan2));
	const copy_plan invalid_plan3{valid_spec, {device_id::d1, {0, 0, 512, 2}, device_id::d2, valid_layout}}; // wrong layout
	CHECK(!is_valid(invalid_plan3));
}

TEST_CASE("copy set validation", "[validation]") {
	const data_layout valid_layout{0, 0, 1024, 1, 1024};
	const copy_spec valid_spec{device_id::d0, valid_layout, device_id::d1, valid_layout};
	const copy_plan valid_plan{valid_spec, {device_id::d1, valid_layout, device_id::d2, valid_layout}};
	const parallel_copy_set valid_set{valid_plan};
	CHECK(is_valid(valid_set));
	const copy_plan invalid_plan{valid_spec, {device_id::d1, valid_layout, device_id::d2, {0, 0, 1024, 1, 512}}};
	const parallel_copy_set invalid_set{invalid_plan};
	CHECK(!is_valid(invalid_set));
}

TEST_CASE("copy plan equivalence", "[equivalence]") {
	const data_layout valid_layout{0, 0, 1024, 1, 1024};
	const copy_spec valid_spec{device_id::d0, valid_layout, device_id::d1, valid_layout};
	const copy_plan trivial_plan{valid_spec};
	CHECK(is_equivalent(trivial_plan, valid_spec));

	// stupid but valid plan
	const copy_plan valid_plan{
	    valid_spec, {device_id::d1, valid_layout, device_id::d2, valid_layout}, {device_id::d2, valid_layout, device_id::d1, valid_layout}};
	CHECK(is_equivalent(valid_plan, valid_spec));

	// plan that doesn't go to the same device
	const copy_plan invalid_plan{{device_id::d0, valid_layout, device_id::d2, valid_layout}};
	CHECK(!is_equivalent(invalid_plan, valid_spec));
}

TEST_CASE("copy set equivalence", "[equivalence]") {
	// set that fully covers the source and target
	const data_layout valid_layout{0, 0, 1024, 1, 1024};
	const copy_spec full_spec{device_id::d0, valid_layout, device_id::d1, valid_layout};
	const data_layout first_half{0, 0, 512, 1, 512};
	const data_layout second_half{0, 512, 512, 1, 512};
	const copy_spec first_half_spec{device_id::d0, first_half, device_id::d1, first_half};
	const copy_spec second_half_spec{device_id::d0, second_half, device_id::d1, second_half};
	const parallel_copy_set full_set{{first_half_spec}, {second_half_spec}};
	CHECK(is_equivalent(full_set, full_spec));
	CHECK(!is_equivalent(full_set, first_half_spec));
	const parallel_copy_set first_half_set{{first_half_spec}};
	CHECK(!is_equivalent(first_half_set, full_spec));
}

TEST_CASE("data layout normalization", "[normalization]") {
	const data_layout contiguous_layout{0, 0, 1024, 1, 1024};
	CHECK(normalize(contiguous_layout) == contiguous_layout);
	CHECK(normalize({0, 0, 512, 2, 512}) == contiguous_layout);
	CHECK(normalize({0, 0, 256, 4, 256}) == contiguous_layout);
	const data_layout non_contiguous_layout{0, 0, 128, 2, 512};
	CHECK(normalize(non_contiguous_layout) == non_contiguous_layout);
}

TEST_CASE("copy spec normalization", "[normalization]") {
	const data_layout contiguous_layout{0, 0, 1024, 1, 1024};
	const copy_spec contiguous_spec{device_id::d0, contiguous_layout, device_id::d1, contiguous_layout};
	CHECK(normalize(contiguous_spec) == contiguous_spec);
	const copy_spec contiguous_multi_fragment_spec{device_id::d0, {0, 0, 512, 2, 512}, device_id::d1, contiguous_layout};
	CHECK(normalize(contiguous_multi_fragment_spec) == contiguous_spec);
	const copy_spec non_contiguous_spec{device_id::d0, {0, 0, 128, 2, 512}, device_id::d1, contiguous_layout};
	CHECK(normalize(non_contiguous_spec) == non_contiguous_spec);
}

TEST_CASE("chunking 1D operations", "[chunking]") {
	constexpr int64_t extra_source_offset = 42;
	const data_layout source{0, extra_source_offset, 1024, 1, 1024};
	const data_layout target{0, 0, 1024, 1, 1024};
	REQUIRE(source.unit_stride());
	REQUIRE(target.unit_stride());

	const copy_spec spec{device_id::d0, source, device_id::d1, target};
	SECTION("single contiguous copy") {
		const auto chunk_size = GENERATE(0, 1024);
		const copy_strategy strategy{copy_type::direct, copy_properties::none, chunk_size};

		const auto copy_set = apply_chunking(spec, strategy);
		REQUIRE(copy_set.size() == 1);
		REQUIRE(copy_set.cbegin()->size() == 1);

		const auto single_copy = copy_set.cbegin()->front();
		CHECK(single_copy.source_device == device_id::d0);
		CHECK(single_copy.source_layout == source);
		CHECK(single_copy.target_device == device_id::d1);
		CHECK(single_copy.target_layout == target);
	}
	SECTION("chunking a contiguous copy (perfectly divisible)") {
		const copy_strategy strategy{copy_type::direct, copy_properties::none, 256};

		const auto copy_set = apply_chunking(spec, strategy);

		parallel_copy_set expected_copy_set;
		for(int i = 0; i < 4; i++) {
			const auto source_offset = i * 256 + extra_source_offset;
			const auto target_offset = i * 256;
			expected_copy_set.insert({{device_id::d0, {0, source_offset, 256, 1, 256}, device_id::d1, {0, target_offset, 256, 1, 256}}});
		}
		CHECK(copy_set == expected_copy_set);
	}
	SECTION("chunking a contiguous copy (not perfectly divisible)") {
		const copy_strategy strategy{copy_type::direct, copy_properties::none, 400};

		const auto copy_set = apply_chunking(spec, strategy);

		parallel_copy_set expected_copy_set;
		for(int i = 0; i < 3; i++) {
			const auto source_offset = i * 400 + extra_source_offset;
			const auto target_offset = i * 400;
			const auto fragment_length = std::min(400, 1024 - i * 400);
			expected_copy_set.insert({{                                                 //
			    device_id::d0, {0, source_offset, fragment_length, 1, fragment_length}, //
			    device_id::d1, {0, target_offset, fragment_length, 1, fragment_length}}});
		}
		CHECK(copy_set == expected_copy_set);
	}
}

TEST_CASE("chunking 2D operations, same fragment length", "[chunking]") {
	const data_layout source{0, 0, 8, 64, 32};
	const data_layout target{0, 0, 8, 64, 96};
	REQUIRE(!source.unit_stride());
	REQUIRE(!target.unit_stride());

	const copy_spec spec{device_id::d0, source, device_id::d1, target};

	SECTION("no chunking necessary") {
		const auto chunk_size = GENERATE(0, 8 * 64);
		const copy_strategy strategy{copy_type::direct, copy_properties::none, chunk_size};

		const auto copy_set = apply_chunking(spec, strategy);
		REQUIRE(copy_set.size() == 1);
		REQUIRE(copy_set.cbegin()->size() == 1);

		const auto single_copy = copy_set.cbegin()->front();
		CHECK(single_copy.source_device == device_id::d0);
		CHECK(single_copy.source_layout == source);
		CHECK(single_copy.target_device == device_id::d1);
		CHECK(single_copy.target_layout == target);
	}
	SECTION("chunking a non-unit stride copy (perfectly divisible)") {
		const copy_strategy strategy{copy_type::direct, copy_properties::none, 256};

		const auto copy_set = apply_chunking(spec, strategy);

		parallel_copy_set expected_copy_set;
		for(int i = 0; i < 2; i++) {
			const auto source_offset = i * (256 / 8 * 32);
			const auto source_count = 256 / 8;
			const auto target_offset = i * (256 / 8 * 96);
			const auto target_count = 256 / 8;
			expected_copy_set.insert({{device_id::d0, {0, source_offset, 8, source_count, 32}, device_id::d1, {0, target_offset, 8, target_count, 96}}});
		}
		CHECK(copy_set == expected_copy_set);
	}
	SECTION("chunking a non-unit stride copy (not perfectly divisible)") {
		const copy_strategy strategy{copy_type::direct, copy_properties::none, 177};

		const auto copy_set = apply_chunking(spec, strategy);

		parallel_copy_set expected_copy_set;
		for(int i = 0; i < 3; i++) {
			const auto source_offset = i * (177 / 8 * 32);
			const auto source_count = std::min(177 / 8, 64 - i * 177 / 8);
			const auto target_offset = i * (177 / 8 * 96);
			const auto target_count = std::min(177 / 8, 64 - i * 177 / 8);
			expected_copy_set.insert({{device_id::d0, {0, source_offset, 8, source_count, 32}, device_id::d1, {0, target_offset, 8, target_count, 96}}});
		}
		CHECK(copy_set == expected_copy_set);
	}
}

TEST_CASE("chunking 2D operations, different fragment length", "[chunking]") {
	const data_layout source{0, 0, 8, 64, 32};
	const data_layout target{0, 0, 32, 16, 96};

	const copy_spec spec{device_id::d0, source, device_id::d1, target};

	SECTION("perfectly divisible") {
		const copy_strategy strategy{copy_type::direct, copy_properties::none, 256};
		const auto copy_set = apply_chunking(spec, strategy);

		parallel_copy_set expected_copy_set;
		for(int i = 0; i < 2; i++) {
			const auto source_offset = i * (256 / 8 * 32);
			const auto source_count = 256 / 8;
			const auto target_offset = i * (256 / 32 * 96);
			const auto target_count = 256 / 32;
			expected_copy_set.insert({{device_id::d0, {0, source_offset, 8, source_count, 32}, device_id::d1, {0, target_offset, 32, target_count, 96}}});
		}

		CHECK(copy_set == expected_copy_set);
	}

	SECTION("with remainder") {
		const copy_strategy strategy{copy_type::direct, copy_properties::none, 177};
		const auto copy_set = apply_chunking(spec, strategy);

		parallel_copy_set expected_copy_set;
		for(int i = 0; i < 4; i++) {
			const auto target_offset = i * (177 / 32 * 96);
			const auto target_count = i == 3 ? 1 : 177 / 32;
			const auto fragment_size_multiplier = target.fragment_length / source.fragment_length;
			const auto source_offset = i * (177 / 32 * fragment_size_multiplier * 32);
			const auto source_count = target_count * fragment_size_multiplier;
			expected_copy_set.insert({{device_id::d0, {0, source_offset, 8, source_count, 32}, device_id::d1, {0, target_offset, 32, target_count, 96}}});
		}

		CHECK(copy_set == expected_copy_set);
	}
}

intptr_t test_staging_buffer_provider(device_id id, int64_t) { return 0x42 + 0x100 * static_cast<int>(id); }

TEST_CASE("staging copy specs at the source end", "[staging]") {
	const data_layout source_layout{0, 0, 16, 64, 128};
	const data_layout target_layout{0, 0, 1024};

	SECTION("no staging desired") {
		const copy_spec spec{device_id::d0, source_layout, device_id::d1, target_layout};
		const copy_strategy strategy{copy_type::direct, copy_properties::none, 0};
		const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
		CHECK(copy_plan.size() == 1);
		CHECK(copy_plan.front() == spec);
	}

	SECTION("no staging necessary") {
		const copy_spec spec{device_id::d0, target_layout, device_id::d1, target_layout};
		const auto copy_type = GENERATE(copy_type::direct, copy_type::staged);
		const copy_strategy strategy{copy_type, copy_properties::none, 0};
		const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
		CHECK(copy_plan.size() == 1);
		CHECK(copy_plan.front() == spec);
	}

	SECTION("staging required") {
		const copy_spec spec{device_id::d0, source_layout, device_id::d1, target_layout};
		const copy_properties props = GENERATE(copy_properties::none, copy_properties::use_kernel, copy_properties::use_2D_copy);
		CAPTURE(props);
		const copy_strategy strategy{copy_type::staged, props, 0};
		const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
		CHECK(copy_plan.size() == 2);
		CHECK(copy_plan.front().properties == props);
		CHECK(copy_plan.front().source_device == device_id::d0);
		CHECK(copy_plan.front().source_layout == source_layout);
		CHECK(copy_plan.front().target_device == device_id::d0);
		CHECK(copy_plan.front().target_layout.unit_stride());
		CHECK(copy_plan.front().target_layout.base == 0x42);
		CHECK(copy_plan.back().properties == props);
		CHECK(copy_plan.back().source_device == device_id::d0);
		CHECK(copy_plan.back().source_layout == copy_plan.front().target_layout);
		CHECK(copy_plan.back().target_device == device_id::d1);
		CHECK(copy_plan.back().target_layout == target_layout);
		CHECK(is_equivalent(copy_plan, spec));
	}
}

TEST_CASE("staging copy specs at the target end", "[staging]") {
	const data_layout source_layout{0, 0, 512};
	const data_layout target_layout{0, 0, 8, 64, 77};

	SECTION("no staging desired") {
		const copy_spec spec{device_id::d0, source_layout, device_id::d1, target_layout};
		const copy_strategy strategy{copy_type::direct, copy_properties::none, 0};
		const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
		CHECK(copy_plan.size() == 1);
		CHECK(copy_plan.front() == spec);
	}

	SECTION("no staging necessary") {
		const copy_spec spec{device_id::d0, source_layout, device_id::d1, source_layout};
		const auto copy_type = GENERATE(copy_type::direct, copy_type::staged);
		const copy_strategy strategy{copy_type, copy_properties::none, 0};
		const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
		CHECK(copy_plan.size() == 1);
		CHECK(copy_plan.front() == spec);
	}

	SECTION("staging required") {
		const copy_spec spec{device_id::d0, source_layout, device_id::d1, target_layout};
		const copy_properties props = GENERATE(copy_properties::none, copy_properties::use_kernel, copy_properties::use_2D_copy);
		CAPTURE(props);
		const copy_strategy strategy{copy_type::staged, props, 0};
		const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
		CHECK(copy_plan.size() == 2);
		CHECK(copy_plan.front().properties == props);
		CHECK(copy_plan.front().source_device == device_id::d0);
		CHECK(copy_plan.front().source_layout == source_layout);
		CHECK(copy_plan.front().target_device == device_id::d1);
		CHECK(copy_plan.front().target_layout.unit_stride());
		CHECK(copy_plan.front().target_layout.base == 0x142);
		CHECK(copy_plan.back().properties == props);
		CHECK(copy_plan.back().source_device == device_id::d1);
		CHECK(copy_plan.back().source_layout == copy_plan.front().target_layout);
		CHECK(copy_plan.back().target_device == device_id::d1);
		CHECK(copy_plan.back().target_layout == target_layout);
		CHECK(is_equivalent(copy_plan, spec));
	}
}

TEST_CASE("staging copy specs at both ends", "[staging]") {
	const auto stride = GENERATE(128, 512);
	const auto offset = GENERATE(0, 31337);
	const data_layout layout{0, offset, 32, 16, stride};

	const copy_spec spec{device_id::d0, layout, device_id::d1, layout};
	const copy_properties props = GENERATE(copy_properties::none, copy_properties::use_kernel, copy_properties::use_2D_copy);
	CAPTURE(props);
	const copy_strategy strategy{copy_type::staged, props, 0};
	const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
	CHECK(copy_plan.size() == 3);
	CHECK(copy_plan.front().properties == props);
	CHECK(copy_plan.front().source_device == device_id::d0);
	CHECK(copy_plan.front().source_layout == layout);
	CHECK(copy_plan.front().target_device == device_id::d0);
	CHECK(copy_plan.front().target_layout.unit_stride());
	CHECK(copy_plan.front().target_layout.base == 0x42);
	CHECK(copy_plan[1].properties == props);
	CHECK(copy_plan[1].source_device == device_id::d0);
	CHECK(copy_plan[1].source_layout == copy_plan.front().target_layout);
	CHECK(copy_plan[1].target_device == device_id::d1);
	CHECK(copy_plan[1].target_layout.unit_stride());
	CHECK(copy_plan[1].target_layout.base == 0x142);
	CHECK(copy_plan.back().properties == props);
	CHECK(copy_plan.back().source_device == device_id::d1);
	CHECK(copy_plan.back().source_layout == copy_plan[1].target_layout);
	CHECK(copy_plan.back().target_device == device_id::d1);
	CHECK(copy_plan.back().target_layout == layout);
	CHECK(is_equivalent(copy_plan, spec));
}

TEST_CASE("implementing copy strategies", "[copy]") {
	const data_layout source_layout{0x10000, 0x42, 16, 1024, 4096};
	const data_layout target_layout{0x20000, 0x0, 32, 512, 3084};

	const copy_spec spec{device_id::d0, source_layout, device_id::d1, target_layout};

	const copy_properties props = GENERATE(copy_properties::none, copy_properties::use_kernel, copy_properties::use_2D_copy);

	const auto verify_properties = [&props](const parallel_copy_set& copy_set) {
		for(const auto& plan : copy_set) {
			for(const auto& copy : plan) {
				if(copy.properties != props) return false;
			}
		}
		return true;
	};

	SECTION("direct copy, no chunking") {
		const copy_strategy strategy{copy_type::direct, props, 0};
		const auto copy_set = manifest_strategy(spec, strategy, test_staging_buffer_provider);
		CHECK(is_equivalent(copy_set, spec));
		CHECK(verify_properties(copy_set));
		CHECK(copy_set.size() == 1);
		CHECK(copy_set.cbegin()->size() == 1);
		auto gen_copy = copy_set.cbegin()->front();
		CHECK(gen_copy.source_device == device_id::d0);
		CHECK(gen_copy.source_layout == source_layout);
		CHECK(gen_copy.target_device == device_id::d1);
		CHECK(gen_copy.target_layout == target_layout);
	}

	SECTION("direct copy, with chunking") {
		const copy_strategy strategy{copy_type::direct, props, 512};
		const auto copy_set = manifest_strategy(spec, strategy, test_staging_buffer_provider);
		CHECK(is_equivalent(copy_set, spec));
		CHECK(verify_properties(copy_set));
		for(const auto& plan : copy_set) {
			for(const auto& copy : plan) {
				CHECK(copy.properties == props);
			}
		}
	}

	SECTION("staged copy, no chunking") {
		const copy_strategy strategy{copy_type::staged, props, 0};
		const auto copy_set = manifest_strategy(spec, strategy, test_staging_buffer_provider);
		CHECK(is_equivalent(copy_set, spec));
		CHECK(copy_set.size() == 1);
		CHECK(verify_properties(copy_set));
	}

	SECTION("staged copy, with chunking") {
		const copy_strategy strategy{copy_type::staged, props, 512};
		const auto copy_set = manifest_strategy(spec, strategy, test_staging_buffer_provider);
		CHECK(is_equivalent(copy_set, spec));
		CHECK(copy_set.size() == 16 * 1024 / 512);
		CHECK(verify_properties(copy_set));
	}

	SECTION("staged copy, with chunking, remainder") {
		const copy_strategy strategy{copy_type::staged, props, 177};
		const auto copy_set = manifest_strategy(spec, strategy, test_staging_buffer_provider);
		CHECK(is_equivalent(copy_set, spec));
		CHECK(copy_set.size() == (16 * 1024) / ((177 / 32) * 32) + 1);
		CHECK(verify_properties(copy_set));
	}
}

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
		if(!valid) { printf("Mismatch at index %ld: expected %u, got %u\n", idx[0], expected, ptr[idx[0]]); }
	});
	queue.wait_and_throw();
	bool valid = *valid_ptr;
	sycl::free(valid_ptr, queue);
	return valid;
}

TEST_CASE("basic copies can be executed", "[executor]") {
	const int64_t buffer_size = 76; // GENERATE(1024, 76);
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

	const copy_properties props = is_2d_copy_available() ? GENERATE(copy_properties::none, copy_properties::use_kernel, copy_properties::use_2D_copy)
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
				if(!valid) { printf("Mismatch at index %ld (byte#%3ld): expected %u, got %u\n", id, elem_idx_byte, expected, ptr[idx[0]]); }
			} else {
				const uint32_t expected = (66 << 24 | 66 << 16 | 66 << 8 | 66);
				const uint32_t valid = ptr[id] == expected;
				sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*valid_ptr).fetch_and(valid);
				if(!valid) { printf("Mismatch at index %ld: expected %u, got %u\n", id, expected, ptr[idx[0]]); }
			}
		});
		queue.wait_and_throw();

		bool valid = *valid_ptr;
		sycl::free(valid_ptr, queue);
		CHECK(valid);
	}
}