cmake_minimum_required(VERSION 3.5.1)
enable_language(CXX)

#--------------------------------------- project
project (SageCal)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 6)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION_REVISION 1)
set(PROJECT_VERSION
  "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

message(STATUS "\n############################\n#   HOST SYSTEM v${CMAKE_HOST_SYSTEM_NAME}\n############################")
message(STATUS "\n############################\n#   Building SageCal v${PROJECT_VERSION}\n############################")

include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)
include(CheckFunctionExists)


#--------------------------------------- paths
SET(MAINFOLDER ${CMAKE_BINARY_DIR})

SET(EXECUTABLE_OUTPUT_PATH "${MAINFOLDER}/dist/bin")
SET(LIBRARY_OUTPUT_PATH "${MAINFOLDER}/dist/lib")

# Load CMAKE configuration from environment variables
set( CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH} )
set( CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH} )


#--------------------------------------- cmake modules
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/CMakeModules)


#--------------------------------------- user options
# TODO: check MPI and CUDA flags!
option (ENABLE_MPI "Enable MPI support" OFF)
option (ENABLE_CUDA "Enable CUDA support" OFF)


#--------------------------------------- rpath setup for libraries

SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

# libsynthesis has mix of C++ and Fortran which is not handled well by
# versions before 2.8.
if(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 2.8)
  if(NOT LIB_EXTRA_SYNTHESIS)
    set(LIB_EXTRA_SYNTHESIS gfortran)
  endif(NOT LIB_EXTRA_SYNTHESIS)
endif(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 2.8)



#--------------------------------------- SageCal dependencies

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

message(STATUS "\n############################\n#   Searching Sagecal dependencies\n############################")
# casacore
find_package(CasaCore REQUIRED)
include_directories(${CASACORE_INCLUDE_DIR})

#cfitsio
find_package(CfitsIO REQUIRED)
include_directories(${CFITSIO_INCLUDE_DIR})

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

# FIXME: is this really needed?
##gfortran
#find_package(GFortranLibs REQUIRED)
#include_directories(${GFORTRAN_INCLUDE_DIR})

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
#set (CMAKE_CXX_FLAGS "-g -O3 -Wall -Wextra -std=c++11 -W -pedantic -ansi -fPIC -fpermissive -fno-omit-frame-pointer -fno-inline-functions -fno-inline-functions-called-once -fno-optimize-sibling-calls")
# set (CMAKE_CXX_FLAGS "-std=c++0x -g -O3 -Wall -Wextra -W -pedantic -ansi -fPIC -fpermissive")
set (CMAKE_CXX_FLAGS "-std=c++0x -g -O3 -Wall")
# FIXME: add debug build option
#set (CMAKE_CXX_FLAGS_DEBUG "-g -pg -pedantic -Wall -DNDEBUG -Wextra -Wcast-align -Wpointer-arith -Woverloaded-virtual -Wwrite-strings -Wcast-qual -Wctor-dtor-privacy -Wno-long-long-Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused")




#--------------------------------------- summary
message(STATUS "\n############################\n#   Configuration summary\n############################")
message (STATUS "CMAKE_SYSTEM .......... = ${CMAKE_SYSTEM}")
message (STATUS "CMAKE_BUILD_TYPE ...... = ${CMAKE_BUILD_TYPE}")
message (STATUS "BUILD_SHARED_LIBS ..... = ${BUILD_SHARED_LIBS}")
message (STATUS "CMAKE_INSTALL_NAME_DIR  = ${CMAKE_INSTALL_NAME_DIR}")
message (STATUS "USE_STACKTRACE ........ = ${USE_STACKTRACE}")
message (STATUS "CMAKE_CXX_COMPILER .... = ${CMAKE_CXX_COMPILER}")
message (STATUS "CMAKE_CXX_FLAGS ....... = ${CMAKE_CXX_FLAGS}")

message (STATUS "ENABLE_MPI ............ = ${ENABLE_MPI}")
message (STATUS "ENABLE_CUDA ........... = ${ENABLE_CUDA}")

message (STATUS "CASACORE_INCLUDE_DIR........... = ${CASACORE_INCLUDE_DIR}")
message (STATUS "CASACORE_LIBRARIES........... = ${CASACORE_LIBRARIES}")

message (STATUS "OpenBLAS_LIB .......... = ${OpenBLAS_LIB}")

message (STATUS "GLIB_PKG_INCLUDE_DIRS.............. = ${GLIB_PKG_INCLUDE_DIRS}")
message (STATUS "GLIB_PKG_LIBRARIES............. = ${GLIB_PKG_LIBRARIES}")

#message (STATUS "LAPACK-INC ............ = ${LAPACK_INCLUDE_DIR}")
message (STATUS "LAPACK_LIBRARIES........... = ${LAPACK_LIBRARIES}")

#message (STATUS "GFORTRAN-INC  ......... = ${GFORTRAN_INCLUDE_DIR}")
#message (STATUS "GFORTRAN-LIBS ......... = ${LIBGFORTRAN_LIBRARIES}")

message (STATUS "CFITSIO_INCLUDE_DIR.......... = ${CFITSIO_INCLUDE_DIR}")
message (STATUS "CFITSIO_LIBRARIES.......... = ${CFITSIO_LIBRARIES}")

message (STATUS "WCSLIB_INCLUDE_DIR........... = ${WCSLIB_INCLUDE_DIR}")
message (STATUS "WCSLIB_LIBRARIES ........... = ${WCSLIB_LIBRARIES}")

message (STATUS "HDF5_INCLUDE_DIR........... = ${HDF5_INCLUDE_DIRS}")
message (STATUS "HDF5_LIBRARIES........... = ${HDF5_LIBRARIES}")



#--------------------------------------- include directories
add_subdirectory(src)


#--------------------------------------- Copy test folder

# FIXME: this will be the final step for testing
#file(COPY ${PROJECT_SOURCE_DIR}/test DESTINATION ${MAINFOLDER}/dist/test)

