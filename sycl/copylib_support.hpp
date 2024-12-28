#pragma once

#include "copylib_core.hpp"

#include <format>

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

// make types format-printable (and ostream for Catch2)
template <>
struct formatter<copylib::device_id> : formatter<string> {
	auto format(const copylib::device_id& p, format_context& ctx) const {
		if(p == copylib::device_id::host) { return formatter<string>::format("host", ctx); }
		return formatter<string>::format(std::format("d{}", static_cast<int>(p)), ctx);
	}
};
template <>
struct formatter<copylib::data_layout> : formatter<string> {
	auto format(const copylib::data_layout& p, format_context& ctx) const {
		std::string addr = (p.is_unplaced_staging()) ? std::format("S{}", p.base) : std::format("{:p}", reinterpret_cast<void*>(p.base));
		return formatter<string>::format(std::format("{{{}+{}, [{} * {}, {}]}}", addr, p.offset, p.fragment_length, p.fragment_count, p.stride), ctx);
	}
};
template <>
struct formatter<copylib::copy_properties> : formatter<string> {
	auto format(const copylib::copy_properties& p, format_context& ctx) const {
		string result;
		if(p & copylib::copy_properties::use_kernel) { result += "use_kernel"; }
		if(p & copylib::copy_properties::use_2D_copy) { result += (result.empty() ? ""s : ","s) + "use_2D_copy"; }
		return formatter<string>::format(result, ctx);
	}
};
template <>
struct formatter<copylib::copy_spec> : formatter<string> {
	auto format(const copylib::copy_spec& p, format_context& ctx) const {
		auto prop_string = ""s;
		if(p.properties != copylib::copy_properties::none) { prop_string = std::format(" ({})", p.properties); }
		return formatter<string>::format(
		    std::format("copy({}{}, {}{}{})", p.source_device, p.source_layout, p.target_device, p.target_layout, prop_string), ctx);
	}
};
template <>
struct formatter<copylib::copy_type> : formatter<string> {
	auto format(const copylib::copy_type& p, format_context& ctx) const {
		switch(p) {
		case copylib::copy_type::direct: return formatter<string>::format("direct", ctx);
		case copylib::copy_type::staged: return formatter<string>::format("staged", ctx);
		default: COPYLIB_ERROR("Unknown copy type {}", static_cast<int>(p));
		}
	}
};
template <>
struct formatter<copylib::copy_strategy> : formatter<string> {
	auto format(const copylib::copy_strategy& p, format_context& ctx) const {
		return formatter<string>::format(std::format("strategy({}({}), chunk:{})", p.type, p.properties, p.chunk_size), ctx);
	}
};
template <>
struct formatter<copylib::copy_plan> : formatter<string> {
	auto format(const copylib::copy_plan& p, format_context& ctx) const {
		ctx.advance_to(std::format_to(ctx.out(), "["));
		for(const auto& spec : p) {
			ctx.advance_to(std::formatter<copylib::copy_spec>{}.format(spec, ctx));
		}
		return std::format_to(ctx.out(), "]");
	}
};
template <>
struct formatter<copylib::parallel_copy_set> : formatter<string> {
	auto format(const copylib::parallel_copy_set& p, format_context& ctx) const {
		ctx.advance_to(std::format_to(ctx.out(), "{{"));
		for(const auto& plan : p) {
			ctx.advance_to(std::formatter<copylib::copy_plan>{}.format(plan, ctx));
		}
		return std::format_to(ctx.out(), "}}");
	}
};

ostream& operator<<(ostream& os, const copylib::device_id& p);
ostream& operator<<(ostream& os, const copylib::data_layout& p);
ostream& operator<<(ostream& os, const copylib::copy_properties& p);
ostream& operator<<(ostream& os, const copylib::copy_spec& p);
ostream& operator<<(ostream& os, const copylib::copy_type& p);
ostream& operator<<(ostream& os, const copylib::copy_strategy& p);

} // namespace std
