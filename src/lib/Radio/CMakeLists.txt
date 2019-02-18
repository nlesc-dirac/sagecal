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

if(ENABLE_CUDA)
    message (STATUS "Compiling lib/Radio with CUDA support.")
# FIXME: add libraries only for gpu version
#    FILE(GLOB extra_objects *.cu)

# objects only for gpu version
    set (extra_objects
        predict_withbeam_cuda
        predict_model
    )
    set(objects ${objects} ${extra_objects})

else()
    message (STATUS "Compiling lib/Radio without CUDA support.")
# FIXME: automate this by using regex
#    FILE(GLOB extra_objects *nocuda.c)

# objects only for cpu version
    set (extra_objects
    )
    set(objects ${objects} ${extra_objects})
endif()

message (STATUS "lib/Radio objects --> ${objects}")




include_directories(./)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dirac)
include_directories(${GLIB_PKG_INCLUDE_DIRS})

foreach (object ${objects})
    add_library(${object} ${object}.c)
    SET_TARGET_PROPERTIES(${object} PROPERTIES SOVERSION 0.0)
endforeach (object)

add_library(radio ${objects} Radio.h)
SET_TARGET_PROPERTIES(radio PROPERTIES SOVERSION 0.0)

## disable install of lib* files
##install(TARGETS ${objects} radio DESTINATION lib)

target_link_libraries(radio
${CASACORE_LIBRARIES}
${BLAS_LIBRARIES}
${LAPACK_LIBRARIES}
${WCSLIB_LIBRARIES}
${GLIB_PKG_LIBRARIES}
)

