#pragma once

#include <catch2/catch_test_macros.hpp>

#include "copylib_support.hpp" // IWYU pragma: keep - this is needed for formatting output, IWYU is dumb

namespace Catch {
template <>
struct StringMaker<copylib::copy_spec> {
	static std::string convert(const copylib::copy_spec& value) {
		return copylib::utils::format("\n{}", value); // extra newline for comparison in test output
	}
};
} // namespace Catch
