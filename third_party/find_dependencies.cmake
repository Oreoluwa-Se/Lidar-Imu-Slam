if (DEFINED THIRD_PARTY)
    set(ROBIN_INCLUDE_DIR ${THIRD_PARTY}/robin-map/include)
    message("The path is set to ${ROBIN_INCLUDE_DIR}")
else()
    message("Path variable THIRD_PARTY is not set")
endif()


# Include robin map content
add_library(robin_map INTERFACE)
add_library(tsl::robin_map ALIAS robin_map)
target_include_directories(robin_map INTERFACE "$<BUILD_INTERFACE:${ROBIN_INCLUDE_DIR}>")
list(APPEND headers 
"${ROBIN_INCLUDE_DIR}/tsl/robin_growth_policy.h"
"${ROBIN_INCLUDE_DIR}/tsl/robin_hash.h" 
"${ROBIN_INCLUDE_DIR}/tsl/robin_map.h"
"${ROBIN_INCLUDE_DIR}/tsl/robin_set.h"
)
target_sources(robin_map INTERFACE "$<BUILD_INTERFACE:${headers}>")


