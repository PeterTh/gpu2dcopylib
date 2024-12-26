#include "copylib.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

namespace Catch {
template <>
struct StringMaker<CopySpec> {
	static std::string convert(const CopySpec& value) {
		return std::format("\n{}", value); // extra newline for comparison in test output
	}
};
} // namespace Catch

TEST_CASE("formatting of types", "[format]") {
	SECTION("DeviceID") {
		CHECK(std::format("{}", DeviceID::host) == "host");
		CHECK(std::format("{}", DeviceID::d0) == "d0");
		CHECK(std::format("{}", DeviceID::d5) == "d5");
	}
	SECTION("DataLayout") {
		const DataLayout layout{nullptr, 0, 1024, 1, 1024};
		CHECK(std::format("{}", layout) == "{0x0+0, [1024 * 1, 1024]}");
	}
	SECTION("CopySpec") {
		const CopySpec spec{DeviceID::d0, {nullptr, 42, 1024, 1, 1024}, DeviceID::d1, {(std::byte*)0xdeadbeef, 0, 256, 4, 320}};
		CHECK(std::format("{}", spec) == "copy(d0{0x0+42, [1024 * 1, 1024]}, d1{0xdeadbeef+0, [256 * 4, 320]})");
	}
}

TEST_CASE("data layout validation", "[validation]") {
	CHECK(is_valid({nullptr, 0, 1024, 1, 1024}));
	CHECK(is_valid({nullptr, 0, 1024, 1, 0}));
	CHECK(!is_valid({nullptr, 0, 1024, 2, 0}));
	CHECK(!is_valid({nullptr, 0, 0, 0, 0}));
	CHECK(!is_valid({nullptr, 0, 1024, 1, 512}));
}

TEST_CASE("copy spec validation", "[validation]") {
	const DataLayout valid_layout{nullptr, 0, 1024, 1, 1024};
	CHECK(is_valid({DeviceID::d0, valid_layout, DeviceID::d1, valid_layout}));
	CHECK(is_valid({DeviceID::d0, valid_layout, DeviceID::d0, {nullptr, 1024, 1024, 1, 1024}}));
	CHECK(!is_valid({DeviceID::d0, valid_layout, DeviceID::d0, valid_layout}));                // overlapping source and target
	CHECK(!is_valid({DeviceID::d0, {nullptr, 0, 0, 1, 1024}, DeviceID::d1, valid_layout}));    // invalid source layout
	CHECK(!is_valid({DeviceID::d0, valid_layout, DeviceID::d1, {nullptr, 0, 1024, 1, 512}}));  // invalid target layout
	CHECK(!is_valid({DeviceID::d0, valid_layout, DeviceID::d1, {nullptr, 0, 2048, 1, 2048}})); // different total bytes
	CHECK(!is_valid({DeviceID::d0, valid_layout, DeviceID::d1, {nullptr, 0, 1024, 2, 1024}})); // different total bytes
	CHECK(is_valid({DeviceID::d0, valid_layout, DeviceID::d1, {nullptr, 256, 512, 2, 512}}));  // fine!
	CHECK(!is_valid({DeviceID::d0, valid_layout, DeviceID::d1, valid_layout, CopyProperties::use_2D_copy | CopyProperties::use_kernel})); // can't have both
	CHECK(is_valid({DeviceID::d0, valid_layout, DeviceID::d1, valid_layout, CopyProperties::use_2D_copy}));                               // fine!
}

TEST_CASE("chunking 1D operations", "[chunking]") {
	constexpr int64_t extra_source_offset = 42;
	const DataLayout source{nullptr, extra_source_offset, 1024, 1, 1024};
	const DataLayout target{nullptr, 0, 1024, 1, 1024};
	REQUIRE(source.unit_stride());
	REQUIRE(target.unit_stride());

	const CopySpec spec{DeviceID::d0, source, DeviceID::d1, target};
	SECTION("single contiguous copy") {
		const auto chunk_size = GENERATE(0, 1024);
		const CopyStrategy strategy{CopyType::direct, CopyProperties::none, chunk_size};

		const auto copy_set = apply_chunking(spec, strategy);
		REQUIRE(copy_set.size() == 1);
		REQUIRE(copy_set.cbegin()->size() == 1);

		const auto single_copy = copy_set.cbegin()->front();
		CHECK(single_copy.source_device == DeviceID::d0);
		CHECK(single_copy.source_layout == source);
		CHECK(single_copy.target_device == DeviceID::d1);
		CHECK(single_copy.target_layout == target);
	}
	SECTION("chunking a contiguous copy (perfectly divisible)") {
		const CopyStrategy strategy{CopyType::direct, CopyProperties::none, 256};

		const auto copy_set = apply_chunking(spec, strategy);

		ParallelCopySet expected_copy_set;
		for(int i = 0; i < 4; i++) {
			const auto source_offset = i * 256 + extra_source_offset;
			const auto target_offset = i * 256;
			expected_copy_set.insert({{DeviceID::d0, {nullptr, source_offset, 256, 1, 256}, DeviceID::d1, {nullptr, target_offset, 256, 1, 256}}});
		}
		CHECK(copy_set == expected_copy_set);
	}
	SECTION("chunking a contiguous copy (not perfectly divisible)") {
		const CopyStrategy strategy{CopyType::direct, CopyProperties::none, 400};

		const auto copy_set = apply_chunking(spec, strategy);

		ParallelCopySet expected_copy_set;
		for(int i = 0; i < 3; i++) {
			const auto source_offset = i * 400 + extra_source_offset;
			const auto target_offset = i * 400;
			const auto fragment_length = std::min(400, 1024 - i * 400);
			expected_copy_set.insert({{                                                      //
			    DeviceID::d0, {nullptr, source_offset, fragment_length, 1, fragment_length}, //
			    DeviceID::d1, {nullptr, target_offset, fragment_length, 1, fragment_length}}});
		}
		CHECK(copy_set == expected_copy_set);
	}
}

TEST_CASE("chunking 2D operations, same fragment length", "[chunking]") {
	const DataLayout source{nullptr, 0, 8, 64, 32};
	const DataLayout target{nullptr, 0, 8, 64, 96};
	REQUIRE(!source.unit_stride());
	REQUIRE(!target.unit_stride());

	const CopySpec spec{DeviceID::d0, source, DeviceID::d1, target};

	SECTION("no chunking necessary") {
		const auto chunk_size = GENERATE(0, 8 * 64);
		const CopyStrategy strategy{CopyType::direct, CopyProperties::none, chunk_size};

		const auto copy_set = apply_chunking(spec, strategy);
		REQUIRE(copy_set.size() == 1);
		REQUIRE(copy_set.cbegin()->size() == 1);

		const auto single_copy = copy_set.cbegin()->front();
		CHECK(single_copy.source_device == DeviceID::d0);
		CHECK(single_copy.source_layout == source);
		CHECK(single_copy.target_device == DeviceID::d1);
		CHECK(single_copy.target_layout == target);
	}
	SECTION("chunking a non-unit stride copy (perfectly divisible)") {
		const CopyStrategy strategy{CopyType::direct, CopyProperties::none, 256};

		const auto copy_set = apply_chunking(spec, strategy);

		ParallelCopySet expected_copy_set;
		for(int i = 0; i < 2; i++) {
			const auto source_offset = i * (256 / 8 * 32);
			const auto source_count = 256 / 8;
			const auto target_offset = i * (256 / 8 * 96);
			const auto target_count = 256 / 8;
			expected_copy_set.insert(
			    {{DeviceID::d0, {nullptr, source_offset, 8, source_count, 32}, DeviceID::d1, {nullptr, target_offset, 8, target_count, 96}}});
		}
		CHECK(copy_set == expected_copy_set);
	}
	SECTION("chunking a non-unit stride copy (not perfectly divisible)") {
		const CopyStrategy strategy{CopyType::direct, CopyProperties::none, 177};

		const auto copy_set = apply_chunking(spec, strategy);

		ParallelCopySet expected_copy_set;
		for(int i = 0; i < 3; i++) {
			const auto source_offset = i * (177 / 8 * 32);
			const auto source_count = std::min(177 / 8, 64 - i * 177 / 8);
			const auto target_offset = i * (177 / 8 * 96);
			const auto target_count = std::min(177 / 8, 64 - i * 177 / 8);
			expected_copy_set.insert(
			    {{DeviceID::d0, {nullptr, source_offset, 8, source_count, 32}, DeviceID::d1, {nullptr, target_offset, 8, target_count, 96}}});
		}
		CHECK(copy_set == expected_copy_set);
	}
}

TEST_CASE("chunking 2D operations, different fragment length", "[chunking]") {
	const DataLayout source{nullptr, 0, 8, 64, 32};
	const DataLayout target{nullptr, 0, 32, 16, 96};

	const CopySpec spec{DeviceID::d0, source, DeviceID::d1, target};

	SECTION("perfectly divisible") {
		const CopyStrategy strategy{CopyType::direct, CopyProperties::none, 256};
		const auto copy_set = apply_chunking(spec, strategy);

		ParallelCopySet expected_copy_set;
		for(int i = 0; i < 2; i++) {
			const auto source_offset = i * (256 / 8 * 32);
			const auto source_count = 256 / 8;
			const auto target_offset = i * (256 / 32 * 96);
			const auto target_count = 256 / 32;
			expected_copy_set.insert(
			    {{DeviceID::d0, {nullptr, source_offset, 8, source_count, 32}, DeviceID::d1, {nullptr, target_offset, 32, target_count, 96}}});
		}

		CHECK(copy_set == expected_copy_set);
	}

	SECTION("with remainder") {
		const CopyStrategy strategy{CopyType::direct, CopyProperties::none, 177};
		const auto copy_set = apply_chunking(spec, strategy);

		ParallelCopySet expected_copy_set;
		for(int i = 0; i < 4; i++) {
			const auto target_offset = i * (177 / 32 * 96);
			const auto target_count = i == 3 ? 1 : 177 / 32;
			const auto fragment_size_multiplier = target.fragment_length / source.fragment_length;
			const auto source_offset = i * (177 / 32 * fragment_size_multiplier * 32);
			const auto source_count = target_count * fragment_size_multiplier;
			expected_copy_set.insert(
			    {{DeviceID::d0, {nullptr, source_offset, 8, source_count, 32}, DeviceID::d1, {nullptr, target_offset, 32, target_count, 96}}});
		}

		CHECK(copy_set == expected_copy_set);
	}
}
