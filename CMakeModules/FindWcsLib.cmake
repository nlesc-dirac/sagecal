# $Id: FindWCSLIB.cmake 13814 2009-08-20 11:55:06Z loose $
#
# Copyright (C) 2008-2009
# ASTRON (Netherlands Foundation for Research in Astronomy)
# P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

# Try to find WCSLIB.
#
# Variables used by this module:
#  WCSLIB_ROOT_DIR     - WCSLIB root directory
#
# Variables defined by this module:
#  WCSLIB_FOUND        - system has WCSLIB
#  WCSLIB_INCLUDE_DIR  - the WCSLIB include directory (cached)
#  WCSLIB_INCLUDE_DIRS - the WCSLIB include directories
#                        (identical to WCSLIB_INCLUDE_DIR)
#  WCSLIB_LIBRARY      - the WCSLIB library (cached)
#  WCSLIB_LIBRARIES    - the WCSLIB libraries
#                        (identical to WCSLIB_LIBRARY)

# find paths
if(NOT WCSLIB_FOUND)

  if (NOT "$ENV{WCSLIB_ROOT_DIR}" STREQUAL "")
  set(WCSLIB_ROOT "$ENV{WCSLIB_ROOT_DIR}" CACHE INTERNAL "Got from environment variable")
  endif()

  find_path(WCSLIB_INCLUDE wcslib/wcs.h
    HINTS ${WCSLIB_ROOT} PATH_SUFFIXES include)
  find_library(WCSLIB_LIB wcs
    HINTS ${WCSLIB_ROOT} PATH_SUFFIXES lib)
  find_library(M_LIBRARY m)
  mark_as_advanced(WCSLIB_INCLUDE WCSLIB_LIB M_LIBRARY)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(WCSLIB DEFAULT_MSG
    WCSLIB_LIB M_LIBRARY WCSLIB_INCLUDE)

  set(WCSLIB_INCLUDE_DIRS ${WCSLIB_INCLUDE})
  set(WCSLIB_LIBRARIES ${WCSLIB_LIB} ${M_LIBRARY})

  set(WCSLIB_INCLUDE ${WCSLIB_INCLUDE})
  set(WCSLIB_LIB ${WCSLIB_LIB} ${M_LIBRARY})

endif(NOT WCSLIB_FOUND)
