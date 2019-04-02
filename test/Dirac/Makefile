CC=gcc
CFLAGS= -Wall -g #-pg
CLIBS= -lm -lpthread

# need BLAS/LAPACK, preferably openblas
LAPACK=-lopenblas -lgfortran -lpthread
LAPACK_DIR=/usr/local/OpenBLAS/lib/

# path to include Dirac.h
INCLUDES= -I../../src/lib/Dirac/
# path to LAPACK and libdirac.a
LIBPATH=-L$(LAPACK_DIR) -L../../build/dist/lib/
# Dirac (Note: when linking, this should be first lib)
LDIRAC=-ldirac

#### glib
GLIBI=-I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/ -I/usr/lib64/glib-2.0/include
GLIBL=-lglib-2.0

OBJECTS=demo.o
OBJECTSSTOCHASTIC=demo_stochastic.o
default:demo
demo.o:demo.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
demo: $(OBJECTS)
	$(CC)  $(CFLAGS) $(INCLUDES) $(GLIBI)  -o $@ $(OBJECTS) $(LDIRAC) $(CLIBS) $(LIBPATH) $(LAPACK)
demo_stochastic.o:demo_stochastic.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
demo_stochastic: $(OBJECTSSTOCHASTIC)
	$(CC)  $(CFLAGS) $(INCLUDES) $(GLIBI)  -o $@ $(OBJECTSSTOCHASTIC) $(LDIRAC) $(CLIBS) $(LIBPATH) $(LAPACK)
