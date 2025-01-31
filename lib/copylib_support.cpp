#include "copylib_support.hpp"

#ifdef COPYLIB_USE_FMT
#define format_to(_a, _b, _c) fmt::format_to(_a, _b, _c)
#endif

namespace std {

#define COPYLIB_OSTREAM_FOR(__type)                                                                                                                            \
	std::ostream& operator<<(std::ostream& os, const copylib::__type& p) {                                                                                     \
		format_to(std::ostreambuf_iterator<char>(os), "{}", p);                                                                                                \
		return os;                                                                                                                                             \
	}

COPYLIB_OSTREAM_FOR(device_id)
COPYLIB_OSTREAM_FOR(staging_id)
COPYLIB_OSTREAM_FOR(data_layout)
COPYLIB_OSTREAM_FOR(copy_properties)
COPYLIB_OSTREAM_FOR(copy_spec)
COPYLIB_OSTREAM_FOR(copy_type)
COPYLIB_OSTREAM_FOR(d2d_implementation)
COPYLIB_OSTREAM_FOR(copy_strategy)
COPYLIB_OSTREAM_FOR(copy_plan)
COPYLIB_OSTREAM_FOR(parallel_copy_set)

#undef COPYLIB_OSTREAM_FOR

} // namespace std
