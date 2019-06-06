find_package(WcsLib REQUIRED)
find_package(FFTW REQUIRED)
include_directories(${WCSLIB_INCLUDE_DIRS}/wcslib)
include_directories(${WCSLIB_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

link_directories(${LIBRARY_OUTPUT_PATH})

FILE(GLOB SRCFILES *.c)
add_executable(restore ${SRCFILES})

target_link_libraries(restore
 ${CFITSIO_LIB}
 ${OpenBLAS_LIB}
 ${LAPACK_LIBRARIES}
 ${WCSLIB_LIBRARIES}
 ${GLIB_PKG_LIBRARIES}
 ${LIBGFORTRAN_LIBRARIES}
 -lfftw3_threads
 -lfftw3
 -lgfortran
 -lpthread
 -lm
)

install(TARGETS restore DESTINATION bin)
