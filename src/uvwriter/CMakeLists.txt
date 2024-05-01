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

link_directories(${LIBRARY_OUTPUT_PATH})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Dirac)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio)

FILE(GLOB SRCFILES *.cpp)

# add build dependencies
target_link_libraries(${target} dirac-radio)
target_link_libraries(${target} dirac)


if (HAVE_CUDA)
  cuda_add_executable(uvwriter ${SRCFILES})
  add_dependencies(uvwriter dirac-radio dirac)
  target_link_libraries(uvwriter
        -ldirac-radio
        -ldirac
        ${CASACORE_LIBRARIES}
        ${CFITSIO_LIB}
        ${OpenBLAS_LIB}
        ${LAPACK_LIBRARIES}
        ${WCSLIB_LIBRARIES}
        ${GLIB_PKG_LIBRARIES}
        $<$<BOOL:${CSPICE_FOUND}>:${CSPICE_LINK_LIBRARIES}>
        -lpthread
        -lm
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        ${CUDA_cusolver_LIBRARIES}
        ${CUDA_cudadevrt_LIBRARIES}
        ${NVML_LIB_PATH}
  )
else()
  add_executable(uvwriter ${SRCFILES})
  add_dependencies(uvwriter dirac-radio dirac)
  target_link_libraries(uvwriter
        -ldirac-radio
        -ldirac
        ${CASACORE_LIBRARIES}
        ${CFITSIO_LIB}
        ${OpenBLAS_LIB}
        ${LAPACK_LIBRARIES}
        ${WCSLIB_LIBRARIES}
        ${GLIB_PKG_LIBRARIES}
        $<$<BOOL:${CSPICE_FOUND}>:${CSPICE_LINK_LIBRARIES}>
        -lpthread
        -lm
  )
endif()
install(TARGETS uvwriter DESTINATION bin)
