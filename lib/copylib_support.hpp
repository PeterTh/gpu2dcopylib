#pragma once

#include "copylib_core.hpp"

#include "utils.hpp"

// this file contains the implementation of exceedingly boring template functions which really shouldn't
// be necessary at all in a modern programming language. Reflection when?

// make types hashable
namespace std {
template <>
struct hash<copylib::data_layout> {
	size_t operator()(const copylib::data_layout& layout) const {
		return copylib::utils::hash_args(layout.base, layout.offset, layout.fragment_length, layout.fragment_count, layout.stride);
	}
};
template <>
struct hash<copylib::copy_properties> {
	size_t operator()(const copylib::copy_properties& prop) const { //
		return hash<int>{}(static_cast<int>(prop));
	}
};
template <>
struct hash<copylib::copy_spec> {
	size_t operator()(const copylib::copy_spec& spec) const {
		return copylib::utils::hash_args(spec.source_device, spec.source_layout, spec.target_device, spec.target_layout, spec.properties);
	}
};
template <>
struct hash<copylib::copy_plan> {
	size_t operator()(const copylib::copy_plan& plan) const {
		size_t val = 0;
		for(const auto& spec : plan) {
			copylib::utils::hash_combine(val, hash<copylib::copy_spec>{}(spec));
		}
		return val;
	}
};
template <>
struct hash<copylib::copy_type> {
	size_t operator()(const copylib::copy_type& type) const { return hash<int>{}(static_cast<int>(type)); }
};
template <>
struct hash<copylib::d2d_implementation> {
	size_t operator()(const copylib::d2d_implementation& impl) const { return hash<int>{}(static_cast<int>(impl)); }
};
template <>
struct hash<copylib::copy_strategy> {
	size_t operator()(const copylib::copy_strategy& strat) const {
		return copylib::utils::hash_args(strat.type, strat.properties, strat.d2d, strat.chunk_size);
	}
};
} // namespace std

#ifdef COPYLIB_USE_FMT
namespace fmt {
#else
namespace std {
#endif

// make types format-printable (and ostream for Catch2)
template <>
struct formatter<copylib::device_id> : formatter<std::string> {
	auto format(const copylib::device_id& p, format_context& ctx) const {
		if(p == copylib::device_id::host) { return formatter<std::string>::format("host", ctx); }
		return formatter<std::string>::format(copylib::utils::format("d{}", static_cast<int>(p)), ctx);
	}
};
template <>
struct formatter<copylib::staging_id> : formatter<std::string> {
	auto format(const copylib::staging_id& p, format_context& ctx) const {
		COPYLIB_ENSURE(p.is_staging_id == copylib::staging_id::staging_id_flag, "Invalid staging id");
		return formatter<std::string>::format(copylib::utils::format("S({}, {}{})", p.index, p.did, p.on_host ? "host" : ""), ctx);
	}
};
template <>
struct formatter<copylib::data_layout> : formatter<std::string> {
	auto format(const copylib::data_layout& p, format_context& ctx) const {
		std::string addr =
		    (p.is_unplaced_staging()) ? copylib::utils::format("{}", p.staging) : copylib::utils::format("{:p}", reinterpret_cast<void*>(p.base));
		return formatter<std::string>::format(
		    copylib::utils::format("{{{}+{}, [{} * {}, {}]}}", addr, p.offset, p.fragment_length, p.fragment_count, p.stride), ctx);
	}
};
template <>
struct formatter<copylib::copy_properties> : formatter<std::string> {
	auto format(const copylib::copy_properties& p, format_context& ctx) const {
		std::string result;
		using namespace std::string_literals;
		if(p & copylib::copy_properties::use_kernel) { result += "use_kernel"; }
		if(p & copylib::copy_properties::use_2D_copy) { result += (result.empty() ? ""s : ","s) + "use_2D_copy"; }
		return formatter<std::string>::format(result, ctx);
	}
};
template <>
struct formatter<copylib::copy_spec> : formatter<std::string> {
	auto format(const copylib::copy_spec& p, format_context& ctx) const {
		using namespace std::string_literals;
		auto prop_string = ""s;
		if(p.properties != copylib::copy_properties::none) { prop_string = copylib::utils::format(" ({})", p.properties); }
		return formatter<std::string>::format(
		    copylib::utils::format("copy({}{}, {}{}{})", p.source_device, p.source_layout, p.target_device, p.target_layout, prop_string), ctx);
	}
};
template <>
struct formatter<copylib::copy_type> : formatter<std::string> {
	auto format(const copylib::copy_type& p, format_context& ctx) const {
		switch(p) {
		case copylib::copy_type::direct: return formatter<std::string>::format("direct", ctx);
		case copylib::copy_type::staged: return formatter<std::string>::format("staged", ctx);
		default: COPYLIB_ERROR("Unknown copy type {}", static_cast<int>(p));
		}
	}
};
template <>
struct formatter<copylib::d2d_implementation> : formatter<std::string> {
	auto format(const copylib::d2d_implementation& p, format_context& ctx) const {
		switch(p) {
		case copylib::d2d_implementation::direct: return formatter<std::string>::format("direct", ctx);
		case copylib::d2d_implementation::host_staging_at_source: return formatter<std::string>::format("host_staging_at_source", ctx);
		case copylib::d2d_implementation::host_staging_at_target: return formatter<std::string>::format("host_staging_at_target", ctx);
		case copylib::d2d_implementation::host_staging_at_both: return formatter<std::string>::format("host_staging_at_both", ctx);
		default: COPYLIB_ERROR("Unknown d2d implementation {}", static_cast<int>(p));
		}
	}
};
template <>
struct formatter<copylib::copy_strategy> : formatter<std::string> {
	auto format(const copylib::copy_strategy& p, format_context& ctx) const {
		return formatter<std::string>::format(copylib::utils::format("strategy({}, {}, d2d:{}, chunk:{})", p.type, p.properties, p.d2d, p.chunk_size), ctx);
	}
};
template <>
struct formatter<copylib::copy_plan> : formatter<std::string> {
	auto format(const copylib::copy_plan& p, format_context& ctx) const {
		ctx.advance_to(format_to(ctx.out(), "["));
		for(size_t i = 0; i < p.size(); i++) {
			const auto& spec = p[i];
			ctx.advance_to(formatter<copylib::copy_spec>{}.format(spec, ctx));
			if(i < p.size() - 1) ctx.advance_to(format_to(ctx.out(), ", "));
		}
		return format_to(ctx.out(), "]");
	}
};
template <>
struct formatter<copylib::parallel_copy_set> : formatter<std::string> {
	auto format(const copylib::parallel_copy_set& p, format_context& ctx) const {
		ctx.advance_to(format_to(ctx.out(), "{{"));
		auto it = p.cbegin();
		for(size_t i = 0; i < p.size(); i++) {
			const auto& plan = *it++;
			ctx.advance_to(formatter<copylib::copy_plan>{}.format(plan, ctx));
			if(i < p.size() - 1) ctx.advance_to(format_to(ctx.out(), ", "));
		}
		return format_to(ctx.out(), "}}");
	}
};
} // namespace fmt // std

namespace std {
ostream& operator<<(ostream& os, const copylib::device_id& p);
ostream& operator<<(ostream& os, const copylib::staging_id& p);
ostream& operator<<(ostream& os, const copylib::data_layout& p);
ostream& operator<<(ostream& os, const copylib::copy_properties& p);
ostream& operator<<(ostream& os, const copylib::copy_spec& p);
ostream& operator<<(ostream& os, const copylib::copy_type& p);
ostream& operator<<(ostream& os, const copylib::d2d_implementation& p);
ostream& operator<<(ostream& os, const copylib::copy_strategy& p);
ostream& operator<<(ostream& os, const copylib::copy_plan& p);
ostream& operator<<(ostream& os, const copylib::parallel_copy_set& p);
} // namespace std
