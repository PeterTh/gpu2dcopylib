#include "utils.hpp"

#include <iostream>
#include <sstream>

#include <sycl/sycl.hpp>

namespace copylib::utils {

void print_to_cerr(std::string_view msg) { //
	std::cerr << msg << std::endl;
}

std::vector<std::string> split(const std::string& str, char delim) {
	std::vector<std::string> result;
	std::string token;
	std::istringstream tokenStream(str);
	while(std::getline(tokenStream, token, delim)) {
		result.push_back(token);
	}
	return result;
}

} // namespace copylib::utils