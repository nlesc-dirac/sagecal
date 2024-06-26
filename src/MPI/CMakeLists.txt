include_directories(${CASACORE_INCLUDE_DIR})
include_directories(${CASACORE_INCLUDE_DIR}/casacore)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Dirac)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio/reserve)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


if(HAVE_CUDA)
    CUDA_INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    CUDA_INCLUDE_DIRECTORIES(${CASACORE_INCLUDE_DIR})
    CUDA_INCLUDE_DIRECTORIES(${CASACORE_INCLUDE_DIR}/casacore)
    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Dirac)
    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio)
    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio/reserve)
    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
endif()

find_package(MPI REQUIRED)
link_directories(${LIBRARY_OUTPUT_PATH})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Dirac)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio)
FILE(GLOB SRCFILES *.cpp)
# add build dependencies
target_link_libraries(${target} dirac-radio)
target_link_libraries(${target} dirac)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
string(STRIP "${SRCFILES}" SRCFILES)

if(HAVE_CUDA)
  cuda_add_executable(sagecal-mpi_gpu ${SRCFILES})
  add_dependencies(sagecal-mpi_gpu dirac-radio dirac)
  target_link_libraries(sagecal-mpi_gpu
    -ldirac-radio
    -ldirac
    ${CASACORE_LIBRARIES}
    ${CFITSIO_LIB}
    ${OpenBLAS_LIB}
    ${LAPACK_LIBRARIES}
    ${WCSLIB_LIBRARIES}
    ${GLIB_PKG_LIBRARIES}
    $<$<BOOL:${CSPICE_FOUND}>:${CSPICE_LINK_LIBRARIES}>
    ${MPI_CXX_LIBRARIES}
    ${MPI_CXX_LINK_FLAGS}
    -lpthread
    -lm
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_cudadevrt_LIBRARY}
    ${NVML_LIB_PATH}
  )

  install(TARGETS sagecal-mpi_gpu DESTINATION bin)
else()

    add_executable(sagecal-mpi ${SRCFILES})
    add_dependencies(sagecal-mpi dirac-radio dirac)
    target_link_libraries(sagecal-mpi
    -ldirac-radio
    -ldirac
    ${CASACORE_LIBRARIES}
    ${CFITSIO_LIB}
    ${OpenBLAS_LIB}
    ${LAPACK_LIBRARIES}
    ${WCSLIB_LIBRARIES}
    ${GLIB_PKG_LIBRARIES}
    $<$<BOOL:${CSPICE_FOUND}>:${CSPICE_LINK_LIBRARIES}>
    ${MPI_CXX_LIBRARIES}
    ${MPI_CXX_LINK_FLAGS}
    -lpthread
    -lm
    )

    install(TARGETS sagecal-mpi DESTINATION bin)

endif()
