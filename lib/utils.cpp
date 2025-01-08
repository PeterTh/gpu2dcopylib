#include "utils.hpp"

#include <iostream>
#include <sstream>

#include <sycl/sycl.hpp>

#if COPYLIB_USE_MIMALLOC
// override default new/delete operators to use the mimalloc memory allocator
#include <mimalloc-new-delete.h>
#endif

namespace copylib::utils {

void dump_to_cerr(std::string_view msg) { std::cerr << msg; }
void dump_to_cout(std::string_view msg) { std::cout << msg; }

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