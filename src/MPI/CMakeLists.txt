include_directories(${CASACORE_INCLUDE_DIR}/casacore)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Dirac)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio/reserve)
include_directories(./)


find_package(MPI REQUIRED)
link_directories(${LIBRARY_OUTPUT_PATH})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Dirac)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio)


FILE(GLOB SRCFILES *.cpp)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
add_executable(sagecal-mpi ${SRCFILES})

target_link_libraries(sagecal-mpi
 ${CASACORE_LIBRARIES}
 ${CFITSIO_LIB}
 ${OpenBLAS_LIB}
 ${LAPACK_LIBRARIES}
 ${WCSLIB_LIBRARIES}
 ${GLIB_PKG_LIBRARIES}
 ${LIBGFORTRAN_LIBRARIES}
 ${MPI_CXX_LIBRARIES}
 ${MPI_CXX_LINK_FLAGS}
 -lopenblas
 -lgfortran
 -lpthread
 -lcfitsio
 -lm
 -ldirac
 -lradio
)

install(TARGETS sagecal-mpi DESTINATION bin)

