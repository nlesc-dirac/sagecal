CC=gcc
CXX=g++
CFLAGS= -Wall -O3 -g 
CLIBS= -lm -lcfitsio -lwcs

# WCSLIB (and CFITSIO)
WCSI= -I/usr/include/wcslib
WCSL= -L/usr/lib
WCSLIBS= -lwcs

# LAPACK
#LAPACK=-L/usr/lib/atlas/sse -llapack -lblas
LAPACK=-lopenblas -lpthread -lgfortran
LAPACK_DIR=/usr/local/OpenBLAS/lib/


INCLUDES= -I. $(WCSI)
LIBPATH=-L$(LAPACK_DIR) $(WCSL)

#### glib
GLIBI=-I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/
GLIBL=-lglib-2.0

OBJECTS2=buildsky.o fitpixels.o scluster.o main.o fitmultipixels.o buildmultisky.o cluster.o myblas.o clmfit_nocuda.o hull.o

default:buildsky
buildsky.o:buildsky.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI) $(LMINCLUDE) -c $<
buildmultisky.o:buildmultisky.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI) $(LMINCLUDE) -c $<
fitpixels.o:fitpixels.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI) $(LMINCLUDE) -c $<
fitmultipixels.o:fitmultipixels.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI) $(LMINCLUDE) -c $<
main.o:main.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI) $(LMINCLUDE) -c $<
scluster.o:scluster.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI) $(LMINCLUDE) -c $<
cluster.o:cluster.c cluster.h
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI) $(LMINCLUDE) -c $<
clmfit_nocuda.o:clmfit_nocuda.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI) $(LMINCLUDE) -c $<
myblas.o:myblas.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI) $(LMINCLUDE) -c $<
hull.o:hull.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI) $(LMINCLUDE) -c $<
buildsky:$(OBJECTS2) buildsky.h
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(GLIBI) $(LIBPATH)  -o $@ $(OBJECTS2)   $(LAPACK) -lcfitsio  $(CLIBS) $(GLIBL) 
