message("-------------------------------------")
message (STATUS "Configuring lib/Dirac")

# Common objects
set (objects
    admm_solve
    clmfit
    manifold_average
    mdl
    myblas
    rtr_solve
    rtr_solve_robust_admm
    updatenu
)

if(HAVE_CUDA)
    message (STATUS "Compiling lib/Dirac with CUDA support.")

    # objects only for gpu version
    set (extra_objects
        barrier
        clmfit_fl
        consensus_poly
        load_balance
        rtr_solve_cuda
        rtr_solve_robust_admm_cuda
        rtr_solve_robust_cuda
        robustlm
    )

    set (extra_objects_cuda
        clmfit_cuda
        lbfgs_cuda
        lmfit_cuda
        manifold_fl
        mderiv
        mderiv_fl
        robust
        robust_fl
        oslmfit
	lbfgs_minibatch_cuda
	robust_batchmode_lbfgs_cuda
  lbfgs_multifreq
    )

    set(objects ${objects} ${extra_objects})

    message (STATUS "Extra CUDA objects ... = ${extra_objects_cuda}")

    message (STATUS "lib/Dirac objects --> ${objects}")

    foreach (object ${objects})
        add_library(${object} SHARED ${object}.c)
        SET_TARGET_PROPERTIES(${object} PROPERTIES SOVERSION 0.0)
    endforeach (object)

    foreach (object ${extra_objects_cuda})
        file(GLOB CUDA_SRC_FILE ${object}.*)
        CUDA_ADD_LIBRARY(${object} SHARED ${CUDA_SRC_FILE})
    endforeach (object)

    set(objects ${objects} ${extra_objects_cuda})

    foreach (object ${objects})
        file(GLOB SRC_FILE ${object}.*)
        set(CUDA_SRC_FILES ${CUDA_SRC_FILES} ${SRC_FILE})
    endforeach (object)


    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
    CUDA_ADD_LIBRARY(dirac ${CUDA_SRC_FILES} Dirac.h)

    SET_TARGET_PROPERTIES(dirac PROPERTIES SOVERSION 0.0)
    SET_TARGET_PROPERTIES(dirac PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


else()
    # objects only for cpu version
    set (extra_objects
        lmfit
        consensus_poly
        lbfgs
        robust_batchmode_lbfgs
        robust_lbfgs
        robustlm
        rtr_solve_robust
    )
    set(objects ${objects} ${extra_objects})

    message (STATUS "lib/Dirac objects --> ${objects}")

    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    foreach (object ${objects})
        file(GLOB SRC_FILE ${object}.*)
        add_library(${object} ${SRC_FILE})
        SET_TARGET_PROPERTIES(${object} PROPERTIES SOVERSION 0.0)
    endforeach (object)

    add_library(dirac ${objects} Dirac.h)
    SET_TARGET_PROPERTIES(dirac PROPERTIES SOVERSION 0.0)

endif()

message("-------------------------------------")
