// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_UTIL_EXCEPTION_HPP__
#define __SPPARK_UTIL_EXCEPTION_HPP__

#include <cstdio>
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>

class sppark_error : public std::runtime_error {
    int _code;

    template<typename... Types>
    inline std::string fmt_errno(int errnum, const char* fmt, Types... args)
    {
        const size_t ERRLEN = 48;
        const int prefix_len = std::snprintf(nullptr, 0, fmt, args...);
        if (prefix_len < 0)
            return {};

        std::vector<char> ret(static_cast<size_t>(prefix_len) + ERRLEN + 1, '\0');
        std::snprintf(ret.data(), static_cast<size_t>(prefix_len) + 1, fmt, args...);
        auto errmsg = ret.data() + prefix_len;
#if defined(_WIN32)
        (void)strerror_s(errmsg, ERRLEN, errnum);
#elif defined(_GNU_SOURCE)
        auto errstr = strerror_r(errnum, errmsg, ERRLEN);
        if (errstr != errmsg)
            std::strncpy(errmsg, errstr, ERRLEN - 1);
#else
        (void)strerror_r(errnum, errmsg, ERRLEN);
#endif
        errmsg[ERRLEN - 1] = '\0';
        return {ret.data(), static_cast<size_t>(prefix_len) + std::strlen(errmsg)};
    }

public:
    sppark_error(int err, const std::string& reason) : std::runtime_error{reason}
    {   _code = err;   }
    sppark_error(int err, const char* msg = "") : std::runtime_error{fmt_errno(err, "%s", msg)}
    {   _code = err;   }
    template<typename... Types>
    sppark_error(int err, const char* fmt, Types... args) : std::runtime_error{fmt_errno(err, fmt, args...)}
    {   _code = err;   }
    inline int code() const
    {   return _code;   }
};

template<typename... Types>
inline std::string fmt(const char* fmt, Types... args)
{
    const int len = std::snprintf(nullptr, 0, fmt, args...);
    if (len < 0)
        return {};

    std::vector<char> ret(static_cast<size_t>(len) + 1, '\0');
    std::snprintf(ret.data(), ret.size(), fmt, args...);
    return {ret.data(), static_cast<size_t>(len)};
}

#endif
