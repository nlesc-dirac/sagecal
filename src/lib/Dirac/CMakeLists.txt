message (STATUS "\n\n\nConfiguring lib/Dirac")


# Common objects
set (objects
    admm_solve
    clmfit
    manifold_average
    mdl
    myblas
    robustlm
    rtr_solve_robust_admm
    updatenu
)

if(ENABLE_CUDA)
    message (STATUS "Compiling lib/Dirac with CUDA support.")

    # file(GLOB_RECURSE extra_objects_cuda "*.cu" "*_cuda_*.c" "*_cuda.c")
    # file(GLOB_RECURSE extra_objects_cuda RELATIVE ${CMAKE_SOURCE_DIR} "src/*.c")
    # file(GLOB extra_objects_cuda RELATIVE ${CMAKE_SOURCE_DIR} "src/lib/Dirac/*.cu" "src/lib/Dirac/*_cuda_*.c" "src/lib/Dirac/*_cuda.c")
    # file(GLOB extra_objects_cuda "*.cu" "*_cuda_*.c" "*_cuda.c")

# objects only for gpu version
    set (extra_objects
        lmfit_cuda
        barrier
        clmfit_cuda
        clmfit_fl
        consensus_poly
        lbfgs_cuda
        load_balance
        oslmfit
        rtr_solve_cuda
        rtr_solve_robust_cuda
        rtr_solve_robust_cuda_admm
    )


    set (extra_objects_cuda
        manifold_fl
        mderiv
        mderiv_fl
        robust
        robust_fl
        rtr_solve_cuda
        rtr_solve_robust_cuda
        rtr_solve_robust_admm_cuda
    )


    set(objects ${objects} ${extra_objects})
    # set(objects ${objects} ${extra_objects_cuda})

    message ("")
    message (STATUS "Extra CUDA objects ... = ${extra_objects_cuda}")
    # get_filename_component(extra_objects_name ${extra_objects} NAME)
    message ("")


    message (STATUS "lib/Dirac objects --> ${objects}")

    include_directories(./)

    foreach (object ${objects})
        add_library(${object} SHARED ${object}.c)
        SET_TARGET_PROPERTIES(${object} PROPERTIES SOVERSION 0.0)
    endforeach (object)

    foreach (object ${extra_objects_cuda})
        CUDA_ADD_LIBRARY(${object}_cu SHARED ${object}.cu)
        # FIXME: are these properties necassary for CUDA?
        # SET_TARGET_PROPERTIES(${object} PROPERTIES SOVERSION 0.0)
        # SET_TARGET_PROPERTIES(${object} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endforeach (object)

    add_library(dirac ${objects} Dirac.h)
    SET_TARGET_PROPERTIES(dirac PROPERTIES SOVERSION 0.0)
    SET_TARGET_PROPERTIES(dirac PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


else()
# FIXME: automate this by using regex
#    FILE(GLOB extra_objects *nocuda.c)

# objects only for cpu version
    set (extra_objects
        lmfit
        consensus_poly
        lbfgs
        robust_lbfgs
        robust_batchmode_lbfgs
        rtr_solve
        rtr_solve_robust
    )
    set(objects ${objects} ${extra_objects})


    message (STATUS "lib/Dirac objects --> ${objects}")

    include_directories(./)
    foreach (object ${objects})
        add_library(${object} SHARED ${object}.c)
        SET_TARGET_PROPERTIES(${object} PROPERTIES SOVERSION 0.0)
    endforeach (object)

    add_library(dirac ${objects} Dirac.h)
    SET_TARGET_PROPERTIES(dirac PROPERTIES SOVERSION 0.0)

endif()



## disable install of lib* files
##install(TARGETS ${objects} dirac DESTINATION lib)
