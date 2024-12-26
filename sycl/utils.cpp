#include "utils.hpp"

#include <iostream>

namespace copylib::utils {

void print_to_cerr(std::string_view msg) { //
	std::cerr << msg << std::endl;
}

} // namespace copylib::utils