#include "copylib.hpp" // IWYU pragma: keep

using namespace copylib;

int main(int, char**) {
	executor exec(8);
	utils::print(exec.get_info());
}
