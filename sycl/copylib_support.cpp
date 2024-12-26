#include "copylib_support.hpp"

namespace std {

ostream& operator<<(ostream& os, const copylib::device_id& p) {
	format_to(ostreambuf_iterator<char>(os), "{}", p);
	return os;
}
ostream& operator<<(ostream& os, const copylib::data_layout& p) {
	format_to(ostreambuf_iterator<char>(os), "{}", p);
	return os;
}
ostream& operator<<(ostream& os, const copylib::copy_properties& p) {
	format_to(ostreambuf_iterator<char>(os), "{}", p);
	return os;
}
ostream& operator<<(ostream& os, const copylib::copy_spec& p) {
	format_to(ostreambuf_iterator<char>(os), "{}", p);
	return os;
}
ostream& operator<<(ostream& os, const copylib::copy_type& p) {
	format_to(ostreambuf_iterator<char>(os), "{}", p);
	return os;
}
ostream& operator<<(ostream& os, const copylib::copy_strategy& p) {
	format_to(ostreambuf_iterator<char>(os), "{}", p);
	return os;
}

} // namespace std
