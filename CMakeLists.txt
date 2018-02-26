#cmake_minimum_required(VERSION 2.8)
cmake_minimum_required(VERSION 3.1.3)
enable_language(CXX)

#--------------------------------------- project
project (SageCal)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION_REVISION 1)
set(PROJECT_VERSION
  "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

message(STATUS "\n############################\n#   Building SageCal v${PROJECT_VERSION}\n############################")

include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)
include(CheckFunctionExists)


#--------------------------------------- paths
SET(MAINFOLDER ${PROJECT_SOURCE_DIR})
SET(EXECUTABLE_OUTPUT_PATH "${MAINFOLDER}/dist/bin")
SET(LIBRARY_OUTPUT_PATH "${MAINFOLDER}/dist/lib")

# Load CMAKE configuration from environment variables
set( CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH} )
set( CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH} )


#--------------------------------------- cmake modules
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/CMakeModules)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/findFFTW/")


#--------------------------------------- user options
# TODO: check MPI and CUDA flags!
option (ENABLE_MPI "Enable MPI support" OFF)
option (ENABLE_CUDA "Enable CUDA support" OFF)

# TODO: add MPI option
# if(ENABLE_MPI)
#     set (USE_THREADS YES)
#     find_package (OpenMP)
#     if (OPENMP_FOUND)
#         if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
#             set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopenmp")
#             set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
#         else( )
#             set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#             set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#         endif( )
#     else (OPENMP_FOUND)
#         message(WARNING "Cannot fullfill ENABLE_MPI, compiler does not support it")
#     endif (OPENMP_FOUND)
# endif(ENABLE_MPI)

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
message(STATUS "Found CasaCore: ${xtensor_INCLUDE_DIRS}")
include_directories(${CASACORE_INCLUDE_DIR})

#cfitsio
find_package(CfitsIO REQUIRED)
message(STATUS "Found CfitsIO: ${CFITSIO_INCLUDE_DIR}")
include_directories(${CFITSIO_INCLUDE_DIR})

#lapack
find_package(LAPACK REQUIRED)
message(STATUS "Found LAPACK: ${LAPACK_INCLUDE_DIR}")
include_directories(${LAPACK_INCLUDE_DIR})

#wcslib
find_package(WcsLib REQUIRED)
message(STATUS "Found LAPACK: ${WCSLIB_INCLUDE_DIR}")
include_directories(${WCSLIB_INCLUDE_DIR})

#blas
find_package(BLAS REQUIRED)
message(STATUS "Found BLAS: ${BLAS_INCLUDE_DIR}")
include_directories(${BLAS_INCLUDE_DIR})

#gfortran
find_package(GFortranLibs REQUIRED)
message(STATUS "Found GFORTRAN: ${GFORTRAN_INCLUDE_DIR}")
include_directories(${GFORTRAN_INCLUDE_DIR})

#glib
pkg_check_modules(GLIB_PKG glib-2.0)
if (GLIB_PKG_FOUND)
    message(STATUS "Found GLIB: ${GLIB_PKG_INCLUDE_DIRS}")
    include_directories(${GLIB_PKG_INCLUDE_DIRS})
endif()


#--------------------------------------- summary
message(STATUS "\n############################\n#   Package summary\n############################")
message (STATUS "CMAKE_SYSTEM .......... = ${CMAKE_SYSTEM}")
message (STATUS "CMAKE_BUILD_TYPE ...... = ${CMAKE_BUILD_TYPE}")
message (STATUS "BUILD_SHARED_LIBS ..... = ${BUILD_SHARED_LIBS}")
message (STATUS "CMAKE_INSTALL_NAME_DIR  = ${CMAKE_INSTALL_NAME_DIR}")
message (STATUS "ENABLE_MPI ............ = ${ENABLE_MPI}")
message (STATUS "USE_STACKTRACE ........ = ${USE_STACKTRACE}")
message (STATUS "CMAKE_CXX_COMPILER .... = ${CMAKE_CXX_COMPILER}")
message (STATUS "CMAKE_CXX_FLAGS ....... = ${CMAKE_CXX_FLAGS}")
message (STATUS "BLAS-LIBS ............. = ${BLAS_LIBRARIES}")
message (STATUS "GLIB-INC .............. = ${GLIB_PKG_INCLUDE_DIRS}")
message (STATUS "GLIB-LIBS ............. = ${GLIB_PKG_LIBRARIES}")
message (STATUS "LAPACK-INC ............ = ${LAPACK_INCLUDE_DIR}")
message (STATUS "LAPACK-LIBS ........... = ${LAPACK_LIBRARIES}")
message (STATUS "GFORTRAN-INC  ......... = ${GFORTRAN_INCLUDE_DIR}")
message (STATUS "GFORTRAN-LIBS ......... = ${LIBGFORTRAN_LIBRARIES}")
message (STATUS "CFITSIO-INC  .......... = ${CFITSIO_INCLUDE_DIR}")
message (STATUS "CFITSIO-LIBS .......... = ${CFITSIO_LIBRARIES}")


#--------------------------------------- build parameters for all targets
set (CMAKE_CXX_FLAGS "-Wall -g -pg -Wextra -std=c++11 -W -Wpointer-arith -Woverloaded-virtual -Wwrite-strings -pedantic -Wno-long-long -O3 -ansi -fPIC -fpermissive -fno-omit-frame-pointer -DNDEBUG -fno-inline-functions -fno-inline-functions-called-once -fno-optimize-sibling-calls")


#--------------------------------------- include directories
# FIXME: add sub-cmake files
#add_subdirectory(src)

