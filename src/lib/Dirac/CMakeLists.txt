message (STATUS "\n\n\nConfiguring lib/Dirac")


# Common objects
set (objects
    admm_solve
    clmfit_nocuda
    manifold_average
    mdl
    myblas
    robustlm
    rtr_solve_robust_admm
    updatenu
)

if(ENABLE_CUDA)
    message (STATUS "Compiling lib/Dirac with CUDA support.")
# FIXME: add libraries only for gpu version
#    FILE(GLOB extra_objects *.cu)

# objects only for gpu version
    set (extra_objects
        lmfit
        barrier
        clmfit
        clmfit_fl
        consensus_poly
        lbfgs
        load_balance
        manifold_fl
        mderiv
        mderiv_fl
        oslmfit
        robust
        robust_fl
        rtr_solve_cuda
        rtr_solve_robust_cuda
        rtr_solve_robust_cuda_admm
    )
    set(objects ${objects} ${extra_objects})

else()
    message (STATUS "Compiling lib/Dirac without CUDA support.")
# FIXME: automate this by using regex
#    FILE(GLOB extra_objects *nocuda.c)

# objects only for cpu version
    set (extra_objects
        lmfit_nocuda
        consensus_poly
        lbfgs_nocuda
        robust_lbfgs_nocuda
        rtr_solve
        rtr_solve_robust
    )
    set(objects ${objects} ${extra_objects})
endif()


message (STATUS "lib/Dirac objects --> ${objects}")

include_directories(./)
foreach (object ${objects})
    add_library(${object} SHARED ${object}.c)
endforeach (object)

add_library(dirac ${objects} Dirac.h)
