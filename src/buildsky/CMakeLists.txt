find_package(WcsLib REQUIRED)
include_directories(${WCSLIB_INCLUDE_DIRS}/wcslib)
include_directories(${WCSLIB_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

link_directories(${LIBRARY_OUTPUT_PATH})

FILE(GLOB SRCFILES *.c)
add_executable(buildsky ${SRCFILES})

target_link_libraries(buildsky
 ${CFITSIO_LIB}
 ${OpenBLAS_LIB}
 ${LAPACK_LIBRARIES}
 ${WCSLIB_LIBRARIES}
 ${GLIB_PKG_LIBRARIES}
 ${LIBGFORTRAN_LIBRARIES}
 -lgfortran
 -lpthread
 -lm
)

install(TARGETS buildsky DESTINATION bin)
