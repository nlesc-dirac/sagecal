OUTPUT=
CXX=mpic++
CXXFLAGS=-O3 -Wall -g -std=c++11 #-pg #-fnostack-protector
# CASA_LIBDIR=/opt/casacore/lib
CASA_LIBDIR=-L/cm/shared/package/casacore/v2.4.1-gcc-6.3.0/lib -L/cm/shared/package/cfitsio/3380-gcc-4.9.3/lib -L/cm/shared/package/lapack/3.6.0-gcc-4.9.3/lib64
# CASA_INCDIR=-I/opt/casacore/include/ -I/opt/casacore/include/casacore
CASA_INCDIR=-I/cm/shared/package/casacore/v2.4.1-gcc-6.3.0/include -I/cm/shared/package/casacore/v2.4.1-gcc-6.3.0/include/casacore
CASA_LIBS=-lcasa_casa -lcasa_tables -lcasa_measures -lcasa_ms -lcfitsio
#LAPACK=-llapack -lblas
LAPACK=-lopenblas -lgfortran -lpthread
# LAPACK_DIR=/usr/local/OpenBLAS/lib/
LAPACK_DIR=/cm/shared/package/openblas/0.2.17mt/lib
#LAPACK_DIR=/usr/lib/atlas/sse/

# LDFLAGS=-Wl,--rpath,/usr/local/OpenBLAS/lib/
LDFLAGS=-Wl,--rpath,/cm/shared/package/openblas/0.2.17mt/lib/,--rpath,${CASA_LIBDIR}
#LDFLAGS=-Wl,-t,--rpath,/software/users/lofareor/SW/lib64
# -Wl,--hash-style=both

MY_LIBS=-lm -lradio -ldirac
INCLUDES=-I. -I../lib/Dirac -I../lib/Radio $(CASA_INCDIR) -I/usr/include
LIBPATH=-L$(LAPACK_DIR) -L$(CASA_LIBDIR) -L../lib/Dirac -L../lib/Radio

#### glib
# GLIBI=-I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/
GLIBI=-I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/ -I/usr/lib64/glib-2.0/include
GLIBL=-lglib-2.0


OBJECTS=main.o data.o sagecal_master.o sagecal_slave.o
default:sagecal-mpi
main.o:main.cpp proto.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GLIBI) -c $<
data.o:data.cpp data.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GLIBI) -c $<
sagecal_master.o:sagecal_master.cpp proto.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GLIBI) -c $<
sagecal_slave.o:sagecal_slave.cpp proto.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GLIBI) -c $<
sagecal-mpi:$(OBJECTS) ../lib/Radio/libradio.a ../lib/Dirac/libdirac.a
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(GLIBI) $(LIBPATH)  -o $@  $(OBJECTS) $(MY_LIBS) $(LAPACK) $(CASA_LIBS)  $(GLIBL)
clean:
	rm *.o *.tmp *.fits
