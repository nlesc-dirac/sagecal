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
target_link_libraries(${target} radio)
target_link_libraries(${target} dirac)

if(HAVE_CUDA)

    cuda_add_executable(sagecal_gpu ${SRCFILES})

    target_link_libraries(sagecal_gpu
        -lradio
        -ldirac
        ${CASACORE_LIBRARIES}
        ${CFITSIO_LIB}
        ${OpenBLAS_LIB}
        ${LAPACK_LIBRARIES}
        ${WCSLIB_LIBRARIES}
        ${GLIB_PKG_LIBRARIES}
        ${LIBGFORTRAN_LIBRARIES}
        -lgfortran
        -lpthread
        -lm
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        -lcudart
        -lcublas
        -lcusolver
        -lcudadevrt
        ${NVML_LIB_PATH}
    )

    install(TARGETS sagecal_gpu DESTINATION bin)

else()

    add_executable(sagecal ${SRCFILES})

    target_link_libraries(sagecal
        -lradio
        -ldirac
        ${CASACORE_LIBRARIES}
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

    install(TARGETS sagecal DESTINATION bin)

endif()


