#pragma once

#include "../Aoce.hpp"

namespace aoce {

ACOE_EXPORT MediaSourceType getMediaType(const std::string& str);

ACOE_EXPORT std::string getAvformat(const std::string& uri);

}