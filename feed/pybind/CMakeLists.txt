set(FEED_PYBIND_DEPS pybind python proto_desc memory executor fleet_wrapper box_wrapper 
    pass_builder parallel_executor profiler layer tracer engine scope_pool
dict_plugin fs shell)

if(WITH_PYTHON)
  list(APPEND FEED_PYBIND_DEPS py_func_op)
endif()

set(FEED_PYBIND_SRCS
  expand_api.cc
   )

if(WITH_PYTHON)
  if(WITH_AMD_GPU)
    hip_library(feed_paddle_pybind SRCS ${FEED_PYBIND_SRCS} DEPS ARCHIVE_START ${FEED_PYBIND_DEPS} ARCHIVE_END)
  else()
    cc_library(feed_paddle_pybind SRCS ${FEED_PYBIND_SRCS} DEPS ${FEED_PYBIND_DEPS})
  endif(WITH_AMD_GPU)

  get_property (os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
  target_link_libraries(feed_paddle_pybind ${os_dependency_modules})

endif(WITH_PYTHON)
