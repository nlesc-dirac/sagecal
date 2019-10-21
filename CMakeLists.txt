cmake_minimum_required(VERSION 3.5.1)

#--------------------------------------- project
project (SageCal)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 7)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION_REVISION 1)
set(PROJECT_VERSION
  "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

message(STATUS "\n############################\n# SageCal v${PROJECT_VERSION}\n############################")


#--------------------------------------- languages
enable_language(C)
enable_language(Fortran)


#--------------------------------------- compiler
message(STATUS "\n############################\n# Building SageCal for ${CMAKE_HOST_SYSTEM_NAME} \n############################")

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release" FORCE)
  endif()
endif()

include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)
include(CheckFunctionExists)


#--------------------------------------- paths
SET(MAINFOLDER ${CMAKE_BINARY_DIR})

SET(EXECUTABLE_OUTPUT_PATH "${MAINFOLDER}/dist/bin")
SET(LIBRARY_OUTPUT_PATH "${MAINFOLDER}/dist/lib")

# Load CMAKE configuration from environment variables
set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH} )
set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH} )

#--------------------------------------- cmake modules
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/CMakeModules)


#--------------------------------------- user options
option (ENABLE_MPI "Enable MPI support" OFF)
option (HAVE_CUDA "Enable CUDA support" OFF)
option (CUDA_DEBUG "Enable Debug mode for CUDA" OFF)
option (NUM_GPU "Number of GPUs" OFF)
option (USE_MIC "Enable MIC" OFF)


#--------------------------------------- rpath setup for libraries

SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")


#--------------------------------------- SageCal dependencies

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

message(STATUS "\n############################\n#   Searching Sagecal dependencies\n############################")
# cuda
if(HAVE_CUDA)
    message (STATUS "Compiling SageCal with GPU support.")
    enable_language(CUDA)
    find_package(CUDA QUIET REQUIRED)
    find_package(NVML REQUIRED)
endif()

# casacore
find_package(CasaCore REQUIRED)
include_directories(${CASACORE_INCLUDE_DIR})

#cfitsio
find_package(CfitsIO REQUIRED)
include_directories(${CFITSIO_INCLUDE})

#lapack
find_package(LAPACK REQUIRED)
include_directories(${LAPACK_INCLUDE_DIR})

#wcslib
find_package(WcsLib REQUIRED)
include_directories(${WCSLIB_INCLUDE_DIR})

#OpenBlas
find_package(OpenBLAS REQUIRED)
include_directories(${OpenBLAS_INCLUDE_DIR})

##blas
#find_package(BLAS REQUIRED)
#include_directories(${BLAS_INCLUDE_DIR})

##gfortran
#find_package(GFortranLibs REQUIRED)
# include_directories(${GFORTRAN_INCLUDE_DIR})

##hdf5
#find_package(HDF5 REQUIRED)


#glib
pkg_check_modules(GLIB_PKG glib-2.0)
if (GLIB_PKG_FOUND)
    include_directories(${GLIB_PKG_INCLUDE_DIRS})
endif()


#--------------------------------------- check environment variables

# Check pre-defined environment variables
if (NOT "$ENV{CASACORE_ROOT_DIR}" STREQUAL "")
    set(CASACORE_ROOT_DIR "$ENV{CASACORE_ROOT_DIR}" CACHE INTERNAL "Copied from environment variable")
    message("Using environment variable CASACORE_ROOT_DIR = ${CASACORE_ROOT_DIR}")
endif()

if (NOT "$ENV{CASACORE_INCLUDE}" STREQUAL "")
    set(CASACORE_INCLUDE_DIR "$ENV{CASACORE_INCLUDE}" CACHE INTERNAL "Copied from environment variable")
    message("Using environment variable CASACORE_INCLUDE_DIR = ${CASACORE_INCLUDE_DIR}")
endif()


#--------------------------------------- build parameters for all targets
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg -Wall -DNDEBUG -Wextra -Wcast-align -Wpointer-arith -Woverloaded-virtual -Wwrite-strings -Wcast-qual -Wctor-dtor-privacy -Wno-long-long-Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused")

# set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD 11)

if(HAVE_CUDA)

  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=compute_61 -gencode arch=compute_62,code=compute_62 --ptxas-options=-v")

  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DHAVE_CUDA")

  # definition for preprocessor
  # TODO: check target_add_definition()
  add_definitions(-DHAVE_CUDA)
  if(CUDA_DEBUG)
    add_definitions(-DCUDA_DEBUG)
  endif()
  if(USE_MIC)
    add_definitions(-DUSE_MIC)
  endif()

  if(NUM_GPU)
    message (STATUS "Setting number of GPUs to ${NUM_GPU}")
    add_definitions(-DMAX_GPU_ID=${NUM_GPU})
  endif()

endif()


#--------------------------------------- summary
message(STATUS "\n############################\n#   Configuration summary\n############################")
message (STATUS "CMAKE_SYSTEM .............. = ${CMAKE_SYSTEM}")
message (STATUS "CMAKE_INSTALL_PREFIX ...... = ${CMAKE_INSTALL_PREFIX}")
message (STATUS "CMAKE_BUILD_TYPE .......... = ${CMAKE_BUILD_TYPE}")
# message (STATUS "BUILD_SHARED_LIBS ......... = ${BUILD_SHARED_LIBS}")
# message (STATUS "USE_STACKTRACE ............ = ${USE_STACKTRACE}")
message (STATUS "CMAKE_CXX_COMPILER ........ = ${CMAKE_CXX_COMPILER}")
message (STATUS "CMAKE_CXX_FLAGS_RELEASE ... = ${CMAKE_CXX_FLAGS_RELEASE}")

message (STATUS "ENABLE_MPI ................ = ${ENABLE_MPI}")
message (STATUS "HAVE_CUDA ................. = ${HAVE_CUDA}")

if(HAVE_CUDA)
  message (STATUS "NUM_GPU ................... = ${NUM_GPU}")
  message (STATUS "CUDA_VERSION .............. = ${CUDA_VERSION}")
  message (STATUS "CMAKE_CUDA_COMPILER ....... = ${CMAKE_CUDA_COMPILER}")
  message (STATUS "CMAKE_CUDA_FLAGS .......... = ${CMAKE_CUDA_FLAGS}")
  message (STATUS "CMAKE_CUDA_FLAGS_DEBUG .... = ${CMAKE_CUDA_FLAGS_DEBUG}")
  message (STATUS "CMAKE_CUDA_FLAGS_RELEASE .. = ${CMAKE_CUDA_FLAGS_RELEASE}")

  message (STATUS "CMAKE_CUDA_HOST_COMPILER .. = ${CMAKE_CUDA_HOST_COMPILER}")
  message (STATUS "CUDACXX ................... = ${CUDACXX}")
  message (STATUS "CUDAHOSTCXX ............... = ${CUDAHOSTCXX}")

  message (STATUS "CUDA_TOOLKIT_ROOT_DIR ..... = ${CUDA_TOOLKIT_ROOT_DIR}")
  message (STATUS "CUDA_INCLUDE_DIRS ......... = ${CUDA_INCLUDE_DIRS}")
  message (STATUS "CUDA_LIBRARIES ............ = ${CUDA_LIBRARIES}")
  message (STATUS "CUDA_CUBLAS_LIBRARIES ..... = ${CUDA_CUBLAS_LIBRARIES}")
  message (STATUS "CUDA_NVCC_FLAGS ........... = ${CUDA_NVCC_FLAGS}")
  message (STATUS "NVCFLAGS .................. = ${NVCFLAGS}")
  message (STATUS "CUDA_DEBUG ................ = ${CUDA_DEBUG}")

  message (STATUS "NVML ...................... = ${NVML}")
  message (STATUS "NVML_LIB_PATH ............. = ${NVML_LIB_PATH}")
  message (STATUS "NVML_INCLUDE_PATH ......... = ${NVML_INCLUDE_PATH}")

endif()


message (STATUS "CASACORE_INCLUDE_DIR....... = ${CASACORE_INCLUDE_DIR}")
message (STATUS "CASACORE_LIBRARIES......... = ${CASACORE_LIBRARIES}")

message (STATUS "OpenBLAS_LIB .............. = ${OpenBLAS_LIB}")

message (STATUS "GLIB_PKG_INCLUDE_DIRS...... = ${GLIB_PKG_INCLUDE_DIRS}")
message (STATUS "GLIB_PKG_LIBRARIES......... = ${GLIB_PKG_LIBRARIES}")

#message (STATUS "LAPACK-INC ................ = ${LAPACK_INCLUDE_DIR}")
message (STATUS "LAPACK_LIBRARIES........... = ${LAPACK_LIBRARIES}")

# message (STATUS "GFORTRAN-INC  ............. = ${GFORTRAN_INCLUDE_DIR}")
# message (STATUS "GFORTRAN-LIBS ............. = ${LIBGFORTRAN_LIBRARIES}")
message (STATUS "CMAKE_Fortran_COMPILER..... = ${CMAKE_Fortran_COMPILER}")

message (STATUS "CFITSIO_ROOT_DIR........... = ${CFITSIO_ROOT_DIR}")
message (STATUS "CFITSIO_INCLUDE............ = ${CFITSIO_INCLUDE}")
message (STATUS "CFITSIO_LIB................ = ${CFITSIO_LIB}")

message (STATUS "WCSLIB_INCLUDE_DIRS........ = ${WCSLIB_INCLUDE_DIRS}")
message (STATUS "WCSLIB_LIBRARIES .......... = ${WCSLIB_LIBRARIES}")

# message (STATUS "HDF5_INCLUDE_DIR........... = ${HDF5_INCLUDE_DIRS}")
# message (STATUS "HDF5_LIBRARIES............. = ${HDF5_LIBRARIES}")


#--------------------------------------- include directories
add_subdirectory(src)

#--------------------------------------- install

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dist/ DESTINATION ${CMAKE_INSTALL_PREFIX})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/buildsky/
        DESTINATION bin
        FILES_MATCHING PATTERN "*.py" )
