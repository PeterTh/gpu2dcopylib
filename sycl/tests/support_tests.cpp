#include "copylib.hpp"    // IWYU pragma: keep
#include "test_utils.hpp" // IWYU pragma: keep

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_contains.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace copylib;
using namespace Catch::Matchers;

TEST_CASE("hashing of types", "[support][hash]") {
	SECTION("data_layout") {
		const data_layout a{0, 0, 1024, 1, 1024};
		const data_layout b{0, 0, 1024, 1, 1024};
		CHECK(std::hash<data_layout>{}(a) == std::hash<data_layout>{}(b));
		const data_layout c{0, 0, 512, 1, 1024};
		CHECK(std::hash<data_layout>{}(a) != std::hash<data_layout>{}(c)); // it would still be a valid hash function otherwise, but come on
	}
	SECTION("copy_properties") {
		CHECK(std::hash<copy_properties>{}(copy_properties::none) == std::hash<copy_properties>{}(copy_properties::none));
		CHECK(std::hash<copy_properties>{}(copy_properties::none) != std::hash<copy_properties>{}(copy_properties::use_kernel));
	}
	SECTION("copy_spec") {
		const copy_spec a{device_id::d0, {0, 0, 1024, 1, 1024}, device_id::d1, {0, 0, 1024, 1, 1024}};
		const copy_spec b{device_id::d0, {0, 0, 1024, 1, 1024}, device_id::d1, {0, 0, 1024, 1, 1024}};
		CHECK(std::hash<copy_spec>{}(a) == std::hash<copy_spec>{}(b));
		const copy_spec c{device_id::d0, {0, 0, 512, 1, 1024}, device_id::d1, {0, 0, 1024, 1, 1024}};
		CHECK(std::hash<copy_spec>{}(a) != std::hash<copy_spec>{}(c));
		copy_spec d = b;
		d.source_device = device_id::d1;
		CHECK(std::hash<copy_spec>{}(a) != std::hash<copy_spec>{}(d));
		copy_spec e = b;
		e.properties = copy_properties::use_kernel;
		CHECK(std::hash<copy_spec>{}(a) != std::hash<copy_spec>{}(e));
	}
	SECTION("copy_plan") {
		const copy_spec s{device_id::d0, {0, 0, 1024, 1, 1024}, device_id::d1, {0, 0, 1024, 1, 1024}};
		const copy_plan a{s};
		const copy_plan b{s};
		CHECK(std::hash<copy_plan>{}(a) == std::hash<copy_plan>{}(b));
		const copy_plan c{s, s};
		CHECK(std::hash<copy_plan>{}(a) != std::hash<copy_plan>{}(c));
	}
}

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
		copy_spec spec2 = spec;
		spec2.properties = copy_properties::use_kernel;
		CHECK(std::format("{}", spec2) == "copy(d0{0x0+42, [1024 * 1, 1024]}, d1{0xdeadbeef+0, [256 * 4, 320]} (use_kernel))");
	}
	SECTION("copy_type") {
		CHECK(std::format("{}", copy_type::direct) == "direct");
		CHECK(std::format("{}", copy_type::staged) == "staged");
	}
	SECTION("copy_strategy") {
		const copy_strategy strategy{copy_type::direct, copy_properties::use_kernel, 256};
		CHECK(std::format("{}", strategy) == "strategy(direct(use_kernel), chunk:256)");
	}
	SECTION("copy_plan") {
		const copy_spec spec{device_id::d0, {0, 0, 1024, 1, 1024}, device_id::d1, {0, 0, 1024, 1, 1024}};
		const copy_plan plan{spec, spec};
		CHECK(std::format("{}", plan)
		      == "[copy(d0{0x0+0, [1024 * 1, 1024]}, d1{0x0+0, [1024 * 1, 1024]}), copy(d0{0x0+0, [1024 * 1, 1024]}, d1{0x0+0, [1024 * 1, 1024]})]");
	}
	SECTION("parallel_copy_set") {
		const copy_spec spec{device_id::d0, {0, 0, 1024, 1, 1024}, device_id::d1, {0, 0, 1024, 1, 1024}};
		const copy_plan plan{spec, spec};
		const parallel_copy_set set{{spec}, plan};
		std::string str = std::format("{}", set);
		CHECK_THAT(str, ContainsSubstring(std::format("{}", spec)));
		CHECK_THAT(str, ContainsSubstring(std::format("{}", plan)));
		CHECK_THAT(str, ContainsSubstring(std::format("], [", plan)));
		CHECK_THAT(str, StartsWith("{"));
		CHECK_THAT(str, EndsWith("}"));
	}
}
