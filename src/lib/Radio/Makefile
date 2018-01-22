CC=gcc
CXX=g++
#CFLAGS= -Wall -O3 -g #-pg
CFLAGS= -Wall -O3 -fopt-info-optimized
CLIBS= -lm -lpthread
#LAPACK=-L/usr/lib/atlas/sse -llapack -lblas
#LAPACK=-L/usr/local/GotoBLAS2/lib -lgoto2 -lpthread -lgfortran
LAPACK=-L/usr/local/OpenBLAS/lib/ -lopenblas -lgfortran -lpthread


INCLUDES= -I. -I../Dirac/ 
LIBPATH=

#### glib
GLIBI=-I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/ -I/usr/lib64/glib-2.0/include
GLIBL=-lglib-2.0

OBJECTS=readsky.o predict.o stationbeam.o predict_withbeam.o transforms.o

default:libradio.a
readsky.o:readsky.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
predict.o:predict.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
stationbeam.o:stationbeam.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
predict_withbeam.o:predict_withbeam.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
transforms.o:transforms.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<


RANLIB=ranlib
libradio.a:$(OBJECTS) Radio.h
	ar rv $@ $(OBJECTS); \
	$(RANLIB) $@;
clean:
	rm *.o libradio.a
