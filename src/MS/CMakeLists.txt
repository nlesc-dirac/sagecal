include_directories(${CASACORE_INCLUDE_DIR}/casacore)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Dirac)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio/reserve)
include_directories(./)

#get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#foreach(dir ${dirs})
#  message(STATUS "include directory  --> '${dir}'")
#endforeach()


link_directories(${LIBRARY_OUTPUT_PATH})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Dirac)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio)


FILE(GLOB SRCFILES *.cpp)
add_executable(sagecal ${SRCFILES})

target_link_libraries(sagecal
 ${CASACORE_LIBRARIES}
 ${CFITSIO_LIBRARIES}
 ${BLAS_LIBRARIES}
 ${LAPACK_LIBRARIES}
 ${WCSLIB_LIBRARIES}
 ${GLIB_PKG_LIBRARIES}
 ${LIBGFORTRAN_LIBRARIES}
 -lpthread
 -lcfitsio
 -lgfortran
 -lm
 -ldirac
 -lradio
)
