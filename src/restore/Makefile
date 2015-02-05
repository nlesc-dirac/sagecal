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

# FFTW
FFTW=-lfftw3_threads -lfftw3

INCLUDES= -I. $(WCSI)
LIBPATH=-L$(LAPACK_DIR) $(WCSL)

#### glib
GLIBI=-I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/
GLIBL=-lglib-2.0

OBJECTS2=restore.o glist.o readsky.o fft.o shapelet_lm.o hermite.o
default:restore
restore.o:restore.c
	$(CC) $(CFLAGS) $(INCLUDES) $(SHINCLUDE) -c $<
glist.o:glist.c
	$(CC) $(CFLAGS) $(INCLUDES) $(SHINCLUDE) -c $<
readsky.o:readsky.c
	$(CC) $(CFLAGS) $(INCLUDES) $(SHINCLUDE) -c $<
fft.o:fft.c
	$(CC) $(CFLAGS) $(INCLUDES) $(SHINCLUDE) -c $<
shapelet_lm.o:shapelet_lm.c
	$(CC) $(CFLAGS) $(INCLUDES) $(SHINCLUDE) -c $<
hermite.o:hermite.c
	$(CC) $(CFLAGS) $(INCLUDES) $(SHINCLUDE) -c $<
restore:$(OBJECTS2) restore.h
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI) $(LIBPATH) -o $@ $(OBJECTS2) $(CLIBS)  $(GLIBL) $(SHLIB) $(FFTW) $(LAPACK)
