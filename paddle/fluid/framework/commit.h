#pragma once

#include <string>

namespace paddle {
namespace framework {

static std::string paddle_commit() {
  return "95c1816ec0";
}

static std::string paddle_compile_branch() {
  return "develop";
}

static std::string paddle_version() {
  return "0.0.0";
}

}  // namespace framework
}  // namespace paddle
