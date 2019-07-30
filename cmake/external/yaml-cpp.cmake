# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include (ExternalProject)

IF(NOT ${WITH_CUSTOM_TRAINER})
  return()
ENDIF(NOT ${WITH_CUSTOM_TRAINER})

set(YAML_SOURCES_DIR ${THIRD_PARTY_PATH}/yaml-cpp)
set(YAML_INSTALL_DIR ${THIRD_PARTY_PATH}/install/yaml-cpp)
set(YAML_INCLUDE_DIR "${YAML_INSTALL_DIR}/include" CACHE PATH "yaml include directory." FORCE)

SET(YAML_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

ExternalProject_Add(
    extern_yaml
    GIT_REPOSITORY "https://github.com/jbeder/yaml-cpp"
    GIT_TAG "yaml-cpp-0.6.2"
    PREFIX          ${YAML_SOURCES_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                    -DCMAKE_CXX_FLAGS=${YAML_CMAKE_CXX_FLAGS}
                    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                    -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                    -DCMAKE_INSTALL_PREFIX=${YAML_INSTALL_DIR}
                    -DCMAKE_INSTALL_LIBDIR=${YAML_INSTALL_DIR}/lib
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DBUILD_TESTING=OFF
                    -DYAML_BUILD_TESTS:BOOL=OFF
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${YAML_INSTALL_DIR}
                     -DCMAKE_INSTALL_LIBDIR:PATH=${YAML_INSTALL_DIR}/lib
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)
set(YAML_LIBRARIES "${YAML_INSTALL_DIR}/lib/libyaml-cpp.a")

add_library(yaml-cpp STATIC IMPORTED GLOBAL)
set_property(TARGET yaml-cpp PROPERTY IMPORTED_LOCATION ${YAML_LIBRARIES})

include_directories(${YAML_INCLUDE_DIR})
add_dependencies(yaml-cpp extern_yaml)
