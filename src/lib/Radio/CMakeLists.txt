message (STATUS "\n\n\nConfiguring lib/Radio")

# Common objects
set (objects
    readsky
    predict
    residual
    transforms
    stationbeam
    predict_withbeam
)

if(HAVE_CUDA)
    message (STATUS "Compiling lib/Radio with CUDA support.")
    # FIXME: add libraries only for gpu version
    #    FILE(GLOB extra_objects *.cu)

    # objects only for gpu version
    set (extra_objects
        predict_withbeam_cuda
    )
    set(objects ${objects} ${extra_objects})

    set (extra_objects_cuda
        predict_model
    )

    message (STATUS "\nExtra CUDA objects ... = ${extra_objects_cuda}\n")
    # get_filename_component(extra_objects_name ${extra_objects} NAME)

    foreach (object ${extra_objects_cuda})
        CUDA_ADD_LIBRARY(${object}_cu SHARED ${object}.cu)
        # FIXME: are these properties necassary for CUDA?
        # SET_TARGET_PROPERTIES(${object} PROPERTIES SOVERSION 0.0)
        # SET_TARGET_PROPERTIES(${object} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endforeach (object)


else()

    message (STATUS "Compiling lib/Radio without CUDA support.")
    # FIXME: automate this by using regex
    #    FILE(GLOB extra_objects *nocuda.c)

    # objects only for cpu version
    set (extra_objects
    )
    set(objects ${objects} ${extra_objects})


    include_directories(./)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dirac)
    include_directories(${GLIB_PKG_INCLUDE_DIRS})

    foreach (object ${objects})
        add_library(${object} ${object}.c)
        SET_TARGET_PROPERTIES(${object} PROPERTIES SOVERSION 0.0)
    endforeach (object)

endif()

message (STATUS "lib/Radio objects --> ${objects}")







if(HAVE_CUDA)

    add_library(radio ${objects} Radio.h)
    SET_TARGET_PROPERTIES(radio PROPERTIES SOVERSION 0.0)

    target_link_libraries(radio
    ${CASACORE_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${WCSLIB_LIBRARIES}
    ${GLIB_PKG_LIBRARIES}
    )

else()

    SET_TARGET_PROPERTIES(radio PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

endif()


## disable install of lib* files
##install(TARGETS ${objects} radio DESTINATION lib)


