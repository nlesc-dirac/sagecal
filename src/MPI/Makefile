OUTPUT=
CXX=mpic++
CXXFLAGS=-O3 -Wall -g #-pg #-fnostack-protector
CASA_LIBDIR=/opt/casacore/lib
CASA_INCDIR=/opt/casacore/include/casacore
CASA_LIBS=-lcasa_casa -lcasa_tables -lcasa_measures -lcasa_ms -lcfitsio
#LAPACK=-llapack -lblas
LAPACK=-lopenblas -lgfortran -lpthread
LAPACK_DIR=/usr/local/OpenBLAS/lib/
#LAPACK_DIR=/usr/lib/atlas/sse/

LDFLAGS=-Wl,--rpath,/usr/local/OpenBLAS/lib/
#LDFLAGS=-Wl,-t,--rpath,/software/users/lofareor/SW/lib64
# -Wl,--hash-style=both

MY_LIBS=-lm -lsagecal
INCLUDES=-I. -I./lib -I$(CASA_INCDIR) -I/usr/include
LIBPATH=-L$(LAPACK_DIR) -L$(CASA_LIBDIR)  -L./lib

#### glib
GLIBI=-I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/
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
sagecal-mpi:$(OBJECTS) ./lib/libsagecal.a
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(GLIBI) $(LIBPATH)  -o $@  $(OBJECTS) $(MY_LIBS) $(LAPACK) $(CASA_LIBS)  $(GLIBL)
clean:
	rm *.o *.tmp *.fits
