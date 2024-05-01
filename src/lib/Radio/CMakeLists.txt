message("-------------------------------------")
message (STATUS "Configuring lib/Radio")

# Common objects
set (objects
    readsky
    predict
    residual
    transforms
    stationbeam
    predict_withbeam
    elementbeam
    shapelet
    diffuse_predict
)

if(CSPICE_FOUND)
    set(extra_objects cspice_utils)
    set(objects ${objects} ${extra_objects})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dirac)
include_directories(${GLIB_PKG_INCLUDE_DIRS})

if(HAVE_CUDA)
    message (STATUS "Compiling lib/Radio with CUDA support.")

    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../Dirac)
    CUDA_INCLUDE_DIRECTORIES(${GLIB_PKG_INCLUDE_DIRS})

    # objects only for gpu version
    set (extra_objects
    )

    set (extra_objects_cuda
        predict_model
        predict_withbeam_cuda
    )

    message (STATUS "Extra CUDA objects ... = ${extra_objects_cuda}")

    #foreach (object ${extra_objects_cuda})
    #    file(GLOB CUDA_SRC_FILE ${object}.*)
    #    CUDA_ADD_LIBRARY(${object} SHARED ${CUDA_SRC_FILE})
    #endforeach (object)

    set(objects ${objects} ${extra_objects} ${extra_objects_cuda})


else()
    # objects only for cpu version
    set (extra_objects
    )

  #foreach (object ${objects})
  #      file(GLOB SRC_FILE ${object}.*)
  #      add_library(${object} ${SRC_FILE})
  #      SET_TARGET_PROPERTIES(${object} PROPERTIES SOVERSION 0.0)
  #  endforeach (object)

    set(objects ${objects} ${extra_objects})

endif()

message (STATUS "lib/Radio objects --> ${objects}")


if(HAVE_CUDA)

    foreach (object ${objects})
        file(GLOB SRC_FILE ${object}.*)
        set(CUDA_SRC_FILES ${CUDA_SRC_FILES} ${SRC_FILE})
    endforeach (object)

    CUDA_ADD_LIBRARY(dirac-radio ${CUDA_SRC_FILES} Dirac_radio.h ${CMAKE_CURRENT_SOURCE_DIR}/../Dirac/Dirac_common.h)

    SET_TARGET_PROPERTIES(dirac-radio PROPERTIES SOVERSION 0.0)
    SET_TARGET_PROPERTIES(dirac-radio PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18")
      SET_TARGET_PROPERTIES(dirac-radio PROPERTIES CUDA_ARCHITECTURES native)
    endif()

    target_link_libraries(dirac-radio
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${WCSLIB_LIBRARIES}
    ${GLIB_PKG_LIBRARIES}
    )

else()

  add_library(dirac-radio SHARED ${objects} Dirac_radio.h ${CMAKE_CURRENT_SOURCE_DIR}/../Dirac/Dirac_common.h)
    SET_TARGET_PROPERTIES(dirac-radio PROPERTIES SOVERSION 0.0)

    target_link_libraries(dirac-radio
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${WCSLIB_LIBRARIES}
    ${GLIB_PKG_LIBRARIES}
    )

endif()

message("-------------------------------------")
