cmake_minimum_required(VERSION 3.10)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20")
 cmake_policy(SET CMP0115 OLD)
endif()
#--------------------------------------- project
project (SAGECal)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 8)
set(PROJECT_VERSION_PATCH 4)
set(PROJECT_VERSION_REVISION 0)
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
option (CUDA_MODEL_MAX_F "Max number of channels in shared mem" OFF)
option (NUM_GPU "Number of GPUs" OFF)
option (USE_MIC "Enable MIC" OFF)
option (LIB_ONLY "Only build libdirac" OFF)


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

message(STATUS "\n############################\n#   Searching Sagecal dependencies\n############################")
# cuda
if(HAVE_CUDA)
    message (STATUS "Compiling SageCal with GPU support.")
    enable_language(CUDA)
    find_package(CUDA QUIET REQUIRED)
    find_package(NVML REQUIRED)
endif()

if(NOT LIB_ONLY)
 # casacore
 find_package(CasaCore REQUIRED)
 include_directories(${CASACORE_INCLUDE_DIR})
endif()


# BLAS: use -DBLA_VENDOR=OpenBLAS for OpenBLAS
find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIR})

if ("${BLA_VENDOR}" STREQUAL "OpenBLAS")
  message(STATUS "Using ${BLA_VENDOR}")
  # add preprocessor def
  add_definitions(-DHAVE_OPENBLAS)
else()
  message(STATUS "Using generic BLAS")
endif()


# LAPACK (will also find the generic BLAS)
find_package(LAPACK REQUIRED)
include_directories(${LAPACK_INCLUDE_DIR})

if(NOT LIB_ONLY)
 #wcslib
 find_package(WcsLib REQUIRED)
 include_directories(${WCSLIB_INCLUDE_DIR})
 #cfitsio
 find_package(CfitsIO REQUIRED)
 include_directories(${CFITSIO_INCLUDE})
endif()

#glib
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB_PKG glib-2.0)
if (GLIB_PKG_FOUND)
    include_directories(${GLIB_PKG_INCLUDE_DIRS})
    link_directories(${GLIB_PKG_LIBRARY_DIRS})
endif()

#CSPICE find if -DCSPICE_PREFIX=... given
set(CSPICE_PREFIX
  ""
  CACHE FILEPATH "Path to CSPICE install root")
if(NOT "${CSPICE_PREFIX}" STREQUAL "")
  set(ENV{PKG_CONFIG_PATH} "${CSPICE_PREFIX}/lib/pkgconfig")
  pkg_search_module(CSPICE libcspice)
endif()
if(CSPICE_FOUND)
  message(STATUS "Found CSPICE: ${CSPICE_INCLUDE_DIRS}")
  include_directories(${CSPICE_INCLUDE_DIRS})
  add_definitions(-DHAVE_CSPICE)
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
  # check if -DCUDA_NVCC_FLAGS is defined by user
  # if not, use default settings
  if(NOT CUDA_NVCC_FLAGS)
    set(CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 --ptxas-options=-v")
  endif()
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
    math(EXPR MAX_GPU "${NUM_GPU} - 1")
    message (STATUS "Setting number of GPUs to ${NUM_GPU}")
    add_definitions(-DMAX_GPU_ID=${MAX_GPU})
  endif()

  if(CUDA_MODEL_MAX_F)
    message (STATUS "Setting shared mem for max channels to ${CUDA_MODEL_MAX_F}")
    add_definitions(-DMODEL_MAX_F=${CUDA_MODEL_MAX_F})
  endif()

endif()


#--------------------------------------- summary
message(STATUS "\n############################\n#   Configuration summary\n############################")
message (STATUS "CMAKE_SYSTEM .............. = ${CMAKE_SYSTEM}")
message (STATUS "CMAKE_INSTALL_PREFIX ...... = ${CMAKE_INSTALL_PREFIX}")
message (STATUS "CMAKE_BUILD_TYPE .......... = ${CMAKE_BUILD_TYPE}")
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

message (STATUS "LAPACK_LIBRARIES........... = ${LAPACK_LIBRARIES}")

message (STATUS "CMAKE_Fortran_COMPILER..... = ${CMAKE_Fortran_COMPILER}")

message (STATUS "CFITSIO_ROOT_DIR........... = ${CFITSIO_ROOT_DIR}")
message (STATUS "CFITSIO_INCLUDE............ = ${CFITSIO_INCLUDE}")
message (STATUS "CFITSIO_LIB................ = ${CFITSIO_LIB}")

message (STATUS "WCSLIB_INCLUDE_DIRS........ = ${WCSLIB_INCLUDE_DIRS}")
message (STATUS "WCSLIB_LIBRARIES .......... = ${WCSLIB_LIBRARIES}")

#--------------------------------------- include directories
add_subdirectory(src)

#-------------------------------- Install pkg-config file
configure_file(${CMAKE_SOURCE_DIR}/src/lib/libdirac.pc.in ${CMAKE_BINARY_DIR}/libdirac.pc @ONLY)
install(FILES
  ${CMAKE_BINARY_DIR}/libdirac.pc
  DESTINATION lib/pkgconfig/)

configure_file(${CMAKE_SOURCE_DIR}/src/lib/libdirac.cmake.in
  ${CMAKE_BINARY_DIR}/libdirac.cmake @ONLY)
install(FILES
  ${CMAKE_BINARY_DIR}/libdirac.cmake
  DESTINATION lib/cmake/)
#--------------------------------------- install

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dist/ DESTINATION ${CMAKE_INSTALL_PREFIX})

if (NOT LIB_ONLY)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/buildsky/
        DESTINATION bin
        FILES_MATCHING PATTERN "*.py" )
endif()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/lib/Dirac/
        DESTINATION include
        FILES_MATCHING PATTERN "Dirac*.h" )
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/lib/Radio/
        DESTINATION include
        FILES_MATCHING PATTERN "Dirac*.h" )
