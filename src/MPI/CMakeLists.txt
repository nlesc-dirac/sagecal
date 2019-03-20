if(HAVE_CUDA)

    message (STATUS "Skipping MPI for GPU build.")

else()

    include_directories(${CASACORE_INCLUDE_DIR}/casacore)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Dirac)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio/reserve)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})


    find_package(MPI REQUIRED)
    link_directories(${LIBRARY_OUTPUT_PATH})
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Dirac)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio)


    FILE(GLOB SRCFILES *.cpp)
    # # add build dependencies
    # target_link_libraries(${target} radio)
    # target_link_libraries(${target} dirac)

    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    string(STRIP ${SRCFILES} SRCFILES)
    add_executable(sagecal-mpi ${SRCFILES})

    target_link_libraries(sagecal-mpi
    -lradio
    -ldirac
    ${CASACORE_LIBRARIES}
    ${CFITSIO_LIB}
    ${OpenBLAS_LIB}
    ${LAPACK_LIBRARIES}
    ${WCSLIB_LIBRARIES}
    ${GLIB_PKG_LIBRARIES}
    ${LIBGFORTRAN_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${MPI_CXX_LINK_FLAGS}
    -lgfortran
    -lpthread
    -lm
    )

    install(TARGETS sagecal-mpi DESTINATION bin)

endif()
