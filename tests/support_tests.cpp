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
		CHECK(utils::format("{}", device_id::host) == "host");
		CHECK(utils::format("{}", device_id::d0) == "d0");
		CHECK(utils::format("{}", device_id::d5) == "d5");
	}
	SECTION("staging_id") {
		const staging_id id{true, device_id::d0, 42};
		CHECK(utils::format("{}", id) == "S(42, d0host)");
		const staging_id id2{false, device_id::d1, 0};
		CHECK(utils::format("{}", id2) == "S(0, d1)");
	}
	SECTION("data_layout") {
		const data_layout layout{0, 0, 1024, 1, 1024};
		CHECK(utils::format("{}", layout) == "{0x0+0, [1024 * 1, 1024]}");
		const data_layout staging_layout{staging_id{false, device_id::d0, 0}, 0, 1024};
		CHECK(utils::format("{}", staging_layout) == "{S(0, d0)+0, [1024 * 1, 1024]}");
	}
	SECTION("copy_properties") {
		CHECK(utils::format("{}", copy_properties::none) == "");
		CHECK(utils::format("{}", copy_properties::use_kernel) == "use_kernel");
		CHECK(utils::format("{}", copy_properties::use_2D_copy) == "use_2D_copy");
		CHECK(utils::format("{}", copy_properties::use_kernel | copy_properties::use_2D_copy) == "use_kernel,use_2D_copy");
	}
	SECTION("copy_spec") {
		const copy_spec spec{device_id::d0, {0, 42, 1024, 1, 1024}, device_id::d1, {0xdead0000, 0, 256, 4, 320}};
		CHECK(utils::format("{}", spec) == "copy(d0{0x0+42, [1024 * 1, 1024]}, d1{0xdead0000+0, [256 * 4, 320]})");
		copy_spec spec2 = spec;
		spec2.properties = copy_properties::use_kernel;
		CHECK(utils::format("{}", spec2) == "copy(d0{0x0+42, [1024 * 1, 1024]}, d1{0xdead0000+0, [256 * 4, 320]} (use_kernel))");
	}
	SECTION("copy_type") {
		CHECK(utils::format("{}", copy_type::direct) == "direct");
		CHECK(utils::format("{}", copy_type::staged) == "staged");
	}
	SECTION("d2d_implementation") {
		CHECK(utils::format("{}", d2d_implementation::direct) == "direct");
		CHECK(utils::format("{}", d2d_implementation::host_staging_at_source) == "host_staging_at_source");
		CHECK(utils::format("{}", d2d_implementation::host_staging_at_target) == "host_staging_at_target");
		CHECK(utils::format("{}", d2d_implementation::host_staging_at_both) == "host_staging_at_both");
	}
	SECTION("copy_strategy") {
		const copy_strategy strategy{copy_type::direct, copy_properties::use_kernel, 256};
		CHECK(utils::format("{}", strategy) == "strategy(direct, use_kernel, d2d:direct, chunk:256)");
	}
	SECTION("copy_plan") {
		const copy_spec spec{device_id::d0, {0, 0, 1024, 1, 1024}, device_id::d1, {0, 0, 1024, 1, 1024}};
		const copy_plan plan{spec, spec};
		CHECK(utils::format("{}", plan)
		      == "[copy(d0{0x0+0, [1024 * 1, 1024]}, d1{0x0+0, [1024 * 1, 1024]}), copy(d0{0x0+0, [1024 * 1, 1024]}, d1{0x0+0, [1024 * 1, 1024]})]");
	}
	SECTION("parallel_copy_set") {
		const copy_spec spec{device_id::d0, {0, 0, 1024, 1, 1024}, device_id::d1, {0, 0, 1024, 1, 1024}};
		const copy_plan plan{spec, spec};
		const parallel_copy_set set{{spec}, plan};
		std::string str = utils::format("{}", set);
		CHECK_THAT(str, ContainsSubstring(utils::format("{}", spec)));
		CHECK_THAT(str, ContainsSubstring(utils::format("{}", plan)));
		CHECK_THAT(str, ContainsSubstring(utils::format("], [", plan)));
		CHECK_THAT(str, StartsWith("{"));
		CHECK_THAT(str, EndsWith("}"));
	}
}
