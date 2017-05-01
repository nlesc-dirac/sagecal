OUTPUT=
CXX=g++
CXXFLAGS=-O3 -Wall -g #-pg #-fnostack-protector
CASA_LIBDIR=-L/cm/shared/package/casacore/v2.1.0-gcc-4.9.3/lib -L/cm/shared/package/cfitsio/3380-gcc-4.9.3/lib -L/cm/shared/package/lapack/3.6.0-gcc-4.9.3/lib64
CASA_INCDIR=-I/cm/shared/package/casacore/v2.1.0-g++-4.9.3/include -I/cm/shared/package/casacore/v2.1.0-g++-4.9.3/include/casacore
CASA_LIBS=-lcasa_casa -lcasa_tables -lcasa_measures -lcasa_ms -lcfitsio
#LAPACK=-llapack -lblas
LAPACK=-lopenblas -lgfortran -lpthread
# LAPACK_DIR=/cm/shared/apps/openblas/0.2.8/lib
LAPACK_DIR=/cm/shared/apps/openblas/0.2.15/lib
#LAPACK_DIR=/usr/lib/atlas/sse/

LDFLAGS=-Wl,--rpath,/usr/local/OpenBLAS/lib/,${CASA_LIBDIR}
#LDFLAGS=-Wl,-t,--rpath,/software/users/lofareor/SW/lib64
# -Wl,--hash-style=both

MY_LIBS=-lm -lsolvers -lsagecal
INCLUDES=-I. -I../lib/Radio -I../lib/Solvers $(CASA_INCDIR) -I/usr/include
LIBPATH=-L$(LAPACK_DIR) $(CASA_LIBDIR)  -L../lib/Radio -L../lib/Solvers

#### glib
GLIBI=-I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/ -I/usr/lib64/glib-2.0/include
GLIBL=-lglib-2.0


OBJECTS=main.o data.o
default:sagecal-gpu
main.o:main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GLIBI) -c $<
data.o:data.cpp data.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GLIBI) -c $<
sagecal-gpu:$(OBJECTS) ../lib/Radio/libsagecal-gpu.a ../lib/Solvers/libsolvers-gpu.a 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(GLIBI) $(LIBPATH)  -o $@  $(OBJECTS) $(MY_LIBS) $(LAPACK) $(CASA_LIBS)  $(GLIBL)
clean:
	rm *.o *.tmp *.fits
