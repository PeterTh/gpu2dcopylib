#include "utils.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace copylib::utils;
using namespace std::string_literals;

TEST_CASE("hashing utilities", "[utils]") {
	CHECK(hash_args(0) != hash_args(1));
	CHECK(hash_args(0, 0) != hash_args(0));
	CHECK(hash_args("bla"s, "alb"s) != hash_args("alb"s, "bla"s));
	CHECK(hash_args("bla"s, 42, 13.7) == hash_args("bla"s, 42, 13.7));
}

TEST_CASE("splitting strings", "[utils]") {
	const auto result = split("a,bla,c", ',');
	REQUIRE(result.size() == 3);
	CHECK(result[0] == "a");
	CHECK(result[1] == "bla");
	CHECK(result[2] == "c");
	const auto result2 = split("hello!world!", '!');
	REQUIRE(result2.size() == 2);
	CHECK(result2[0] == "hello");
	CHECK(result2[1] == "world");
}

TEST_CASE("calculating median of vector", "[utils]") {
	CHECK(vector_median(std::vector{1, 2, 5, 4, 3}) == 3);
	CHECK(vector_median(std::vector{1.0, 3.0, 4.0, 5.0, 2.0, 6.0}) == 3.5);
	CHECK(vector_median(std::vector{20, 10, 1, 1, 5}) == 5);
}

TEST_CASE("calculating vector metrics", "[utils]") {
	const auto metrics = vector_metrics(std::vector{4, 2, 3, 1, 5});
	CHECK(metrics.median == 3);
	CHECK(metrics.percentile_25 == 2);
	CHECK(metrics.percentile_75 == 4);
}
