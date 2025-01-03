#include "copylib_core.hpp"
#include "test_utils.hpp" // IWYU pragma: keep

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace copylib;

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
	const copy_plan invalid_plan3{valid_spec, {device_id::d1, {0, 0, 512, 2, 512}, device_id::d2, valid_layout}}; // wrong layout
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
		const copy_strategy strategy{chunk_size};

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
		const copy_strategy strategy{256};

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
		const copy_strategy strategy{400};

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
		const copy_strategy strategy{chunk_size};

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
		const copy_strategy strategy{256};

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
		const copy_strategy strategy{177};

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
		const copy_strategy strategy{256};
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
		const copy_strategy strategy{177};
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

staging_id test_staging_buffer_provider(device_id did, bool on_host, int64_t) { return {on_host, did, 42}; }

TEST_CASE("staging copy specs at the source end", "[staging]") {
	const data_layout source_layout{0, 0, 16, 64, 128};
	const data_layout target_layout{0, 0, 1024};

	SECTION("no staging desired") {
		const copy_spec spec{device_id::d0, source_layout, device_id::d1, target_layout};
		const copy_strategy strategy{copy_type::direct};
		const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
		CHECK(copy_plan.size() == 1);
		CHECK(copy_plan.front() == spec);
	}

	SECTION("no staging necessary") {
		const copy_spec spec{device_id::d0, target_layout, device_id::d1, target_layout};
		const auto copy_type = GENERATE(copy_type::direct, copy_type::staged);
		const copy_strategy strategy{copy_type};
		const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
		CHECK(copy_plan.size() == 1);
		CHECK(copy_plan.front() == spec);
	}

	SECTION("staging required") {
		const std::pair<device_id, device_id> devices = GENERATE(
		    std::make_pair(device_id::d0, device_id::d1), std::make_pair(device_id::host, device_id::d0), std::make_pair(device_id::d0, device_id::host));
		const auto src_dev = devices.first;
		const auto tgt_dev = devices.second;
		CAPTURE(src_dev, tgt_dev);

		const copy_spec spec{src_dev, source_layout, tgt_dev, target_layout};
		const copy_properties props = GENERATE(copy_properties::none, copy_properties::use_kernel);
		CAPTURE(props);

		const copy_strategy strategy{copy_type::staged, props};
		const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
		CHECK(copy_plan.size() == 2);
		CHECK(copy_plan.front().properties == props);
		CHECK(copy_plan.front().source_device == src_dev);
		CHECK(copy_plan.front().source_layout == source_layout);
		CHECK(copy_plan.front().target_device == src_dev);
		CHECK(copy_plan.front().target_layout.unit_stride());
		// staged on the source device, but if the source is host, the staging is on the target device
		const auto staging_device = src_dev == device_id::host ? tgt_dev : src_dev;
		CHECK(copy_plan.front().target_layout.staging == staging_id{src_dev == device_id::host, staging_device, 42});
		CHECK(copy_plan.back().properties == props);
		CHECK(copy_plan.back().source_device == src_dev);
		CHECK(copy_plan.back().source_layout == copy_plan.front().target_layout);
		CHECK(copy_plan.back().target_device == tgt_dev);
		CHECK(copy_plan.back().target_layout == target_layout);
		CHECK(is_equivalent(copy_plan, spec));
	}
}

TEST_CASE("staging copy specs at the target end", "[staging]") {
	const data_layout source_layout{0, 0, 512};
	const data_layout target_layout{0, 0, 8, 64, 77};

	SECTION("no staging desired") {
		const copy_spec spec{device_id::d0, source_layout, device_id::d1, target_layout};
		const copy_strategy strategy{copy_type::direct};
		const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
		CHECK(copy_plan.size() == 1);
		CHECK(copy_plan.front() == spec);
	}

	SECTION("no staging necessary") {
		const copy_spec spec{device_id::d0, source_layout, device_id::d1, source_layout};
		const auto copy_type = GENERATE(copy_type::direct, copy_type::staged);
		const copy_strategy strategy{copy_type};
		const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
		CHECK(copy_plan.size() == 1);
		CHECK(copy_plan.front() == spec);
	}

	SECTION("staging required") {
		const std::pair<device_id, device_id> devices = GENERATE(
		    std::make_pair(device_id::d0, device_id::d1), std::make_pair(device_id::host, device_id::d0), std::make_pair(device_id::d0, device_id::host));
		const auto src_dev = devices.first;
		const auto tgt_dev = devices.second;
		CAPTURE(src_dev, tgt_dev);

		const copy_spec spec{src_dev, source_layout, tgt_dev, target_layout};
		const copy_properties props = GENERATE(copy_properties::none, copy_properties::use_kernel);
		CAPTURE(props);

		const copy_strategy strategy{copy_type::staged, props};
		const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
		CHECK(copy_plan.size() == 2);
		CHECK(copy_plan.front().properties == props);
		CHECK(copy_plan.front().source_device == src_dev);
		CHECK(copy_plan.front().source_layout == source_layout);
		CHECK(copy_plan.front().target_device == tgt_dev);
		CHECK(copy_plan.front().target_layout.unit_stride());
		// staged on the target device, but if the target is host, the staging is on the source device
		const auto staging_device = tgt_dev == device_id::host ? src_dev : tgt_dev;
		CHECK(copy_plan.front().target_layout.staging == staging_id{tgt_dev == device_id::host, staging_device, 42});
		CHECK(copy_plan.back().properties == props);
		CHECK(copy_plan.back().source_device == tgt_dev);
		CHECK(copy_plan.back().source_layout == copy_plan.front().target_layout);
		CHECK(copy_plan.back().target_device == tgt_dev);
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
	const copy_strategy strategy{copy_type::staged, props};
	const auto copy_plan = apply_staging(spec, strategy, test_staging_buffer_provider);
	CHECK(copy_plan.size() == 3);
	CHECK(copy_plan.front().properties == props);
	CHECK(copy_plan.front().source_device == device_id::d0);
	CHECK(copy_plan.front().source_layout == layout);
	CHECK(copy_plan.front().target_device == device_id::d0);
	if(!(props & copy_properties::use_2D_copy)) CHECK(copy_plan.front().target_layout.unit_stride());
	CHECK(copy_plan.front().target_layout.staging == staging_id{false, device_id::d0, 42});
	CHECK(copy_plan[1].properties == props);
	CHECK(copy_plan[1].source_device == device_id::d0);
	CHECK(copy_plan[1].source_layout == copy_plan.front().target_layout);
	CHECK(copy_plan[1].target_device == device_id::d1);
	if(!(props & copy_properties::use_2D_copy)) CHECK(copy_plan[1].target_layout.unit_stride());
	CHECK(copy_plan[1].target_layout.staging == staging_id{false, device_id::d1, 42});
	CHECK(copy_plan.back().properties == props);
	CHECK(copy_plan.back().source_device == device_id::d1);
	CHECK(copy_plan.back().source_layout == copy_plan[1].target_layout);
	CHECK(copy_plan.back().target_device == device_id::d1);
	CHECK(copy_plan.back().target_layout == layout);
	CHECK(is_equivalent(copy_plan, spec));
}

TEST_CASE("Applying d2d implementations", "[d2d]") {
	const data_layout src_layout{0, 0, 16, 64, 128};
	const data_layout tgt_layout = src_layout;
	const copy_spec spec{device_id::d0, src_layout, device_id::d1, tgt_layout};

	SECTION("direct copy") {
		const copy_strategy strategy{copy_type::direct};
		const auto copy_plan = apply_d2d_implementation({spec}, d2d_implementation::direct, test_staging_buffer_provider);
		REQUIRE(copy_plan.size() == 1);
		CHECK(copy_plan.front() == spec);
	}
	SECTION("staged on one end") {
		const copy_strategy strategy{copy_type::staged};
		const d2d_implementation impl = GENERATE(d2d_implementation::host_staging_at_source, d2d_implementation::host_staging_at_target);
		CAPTURE(impl);
		const auto copy_plan = apply_d2d_implementation({spec}, impl, test_staging_buffer_provider);
		REQUIRE(copy_plan.size() == 2);
		const staging_id expected_staging = staging_id{true, impl == d2d_implementation::host_staging_at_source ? device_id::d0 : device_id::d1, 42};
		const data_layout expected_staged_layout = {
		    expected_staging, src_layout.offset, src_layout.fragment_length, src_layout.fragment_count, src_layout.stride};
		CHECK(copy_plan.front() == copy_spec{device_id::d0, src_layout, device_id::host, expected_staged_layout});
		CHECK(copy_plan.back() == copy_spec{device_id::host, expected_staged_layout, device_id::d1, tgt_layout});
		CHECK(is_equivalent(copy_plan, spec));
	}
	SECTION("staged on both ends") {
		const copy_strategy strategy{copy_type::staged};
		const auto copy_plan = apply_d2d_implementation({spec}, d2d_implementation::host_staging_at_both, test_staging_buffer_provider);
		REQUIRE(copy_plan.size() == 3);
		const staging_id staging_id_1{true, device_id::d0, 42};
		const staging_id staging_id_2{true, device_id::d1, 42};
		const data_layout staged_layout_1 = {staging_id_1, src_layout.offset, src_layout.fragment_length, src_layout.fragment_count, src_layout.stride};
		const data_layout staged_layout_2 = {staging_id_2, src_layout.offset, src_layout.fragment_length, src_layout.fragment_count, src_layout.stride};
		CHECK(copy_plan.front() == copy_spec{device_id::d0, src_layout, device_id::host, staged_layout_1});
		CHECK(copy_plan[1] == copy_spec{device_id::host, staged_layout_1, device_id::host, staged_layout_2});
		CHECK(copy_plan.back() == copy_spec{device_id::host, staged_layout_2, device_id::d1, tgt_layout});
		CHECK(is_equivalent(copy_plan, spec));
	}
}

TEST_CASE("Applying d2d implementations to staged plans", "[d2d]") {
	const data_layout src_layout{0, 0, 16, 64, 128};
	const data_layout tgt_layout = src_layout;
	const copy_spec spec{device_id::d0, src_layout, device_id::d1, tgt_layout};
	const auto staged_plan = apply_staging(spec, copy_strategy{copy_type::staged}, test_staging_buffer_provider);

	SECTION("direct copy") {
		const copy_strategy strategy{copy_type::direct};
		const auto copy_plan = apply_d2d_implementation(staged_plan, d2d_implementation::direct, test_staging_buffer_provider);
		REQUIRE(copy_plan.size() == 3);
		CHECK(copy_plan == staged_plan);
	}
	SECTION("staged on one end") {
		const copy_strategy strategy{copy_type::staged};
		const d2d_implementation impl = GENERATE(d2d_implementation::host_staging_at_source, d2d_implementation::host_staging_at_target);
		CAPTURE(impl);
		const auto copy_plan = apply_d2d_implementation(staged_plan, impl, test_staging_buffer_provider);
		REQUIRE(copy_plan.size() == 4);
		CHECK(is_equivalent(copy_plan, spec));
	}
	SECTION("staged on both ends") {
		const copy_strategy strategy{copy_type::staged};
		const auto copy_plan = apply_d2d_implementation(staged_plan, d2d_implementation::host_staging_at_both, test_staging_buffer_provider);
		REQUIRE(copy_plan.size() == 5);
		CHECK(is_equivalent(copy_plan, spec));
	}
}

TEST_CASE("implementing copy strategies", "[copy]") {
	const data_layout source_layout{0x10000, 0x42, 16, 1024, 4096};
	const int frag_size_multiplier = GENERATE(1, 2);
	CAPTURE(frag_size_multiplier);
	const data_layout target_layout{0x20000, 0x0, 16 * frag_size_multiplier, 1024 / frag_size_multiplier, 3084};

	const copy_spec spec{device_id::d0, source_layout, device_id::d1, target_layout};

	const copy_properties props = frag_size_multiplier == 1 ? GENERATE(copy_properties::none, copy_properties::use_kernel, copy_properties::use_2D_copy)
	                                                        : GENERATE(copy_properties::none, copy_properties::use_kernel);
	CAPTURE(props);

	const auto verify_properties = [&props](const parallel_copy_set& copy_set) {
		for(const auto& plan : copy_set) {
			for(const auto& copy : plan) {
				if(copy.properties != props) return false;
			}
		}
		return true;
	};

	SECTION("direct copy, no chunking") {
		const copy_strategy strategy{copy_type::direct, props};
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
		const d2d_implementation impl = GENERATE(d2d_implementation::direct, d2d_implementation::host_staging_at_source,
		    d2d_implementation::host_staging_at_target, d2d_implementation::host_staging_at_both);
		CAPTURE(impl);
		const copy_strategy strategy{copy_type::staged, props, impl};
		const auto copy_set = manifest_strategy(spec, strategy, test_staging_buffer_provider);
		CHECK(is_equivalent(copy_set, spec));
		CHECK(copy_set.size() == 1);
		CHECK(verify_properties(copy_set));
	}

	SECTION("staged copy, with chunking") {
		const d2d_implementation impl = GENERATE(d2d_implementation::direct, d2d_implementation::host_staging_at_source,
		    d2d_implementation::host_staging_at_target, d2d_implementation::host_staging_at_both);
		CAPTURE(impl);
		const copy_strategy strategy{copy_type::staged, props, impl, 512};
		const auto copy_set = manifest_strategy(spec, strategy, test_staging_buffer_provider);
		CHECK(is_equivalent(copy_set, spec));
		CHECK(copy_set.size() == 16 * 1024 / 512);
		CHECK(verify_properties(copy_set));
	}

	SECTION("staged copy, with chunking, remainder") {
		const d2d_implementation impl = GENERATE(d2d_implementation::direct, d2d_implementation::host_staging_at_source,
		    d2d_implementation::host_staging_at_target, d2d_implementation::host_staging_at_both);
		CAPTURE(impl);
		const copy_strategy strategy{copy_type::staged, props, impl, 177};
		const auto copy_set = manifest_strategy(spec, strategy, test_staging_buffer_provider);
		CHECK(is_equivalent(copy_set, spec));
		const auto target_frag_length = target_layout.fragment_length;
		CHECK(static_cast<int64_t>(copy_set.size()) == (16 * 1024) / ((177 / target_frag_length) * target_frag_length) + 1);
		CHECK(verify_properties(copy_set));
	}
}
