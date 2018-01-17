CC=gcc
CXX=g++
CFLAGS= -Wall -g -pg
# Extra args for making callgraphs.
# CFLAGS= -Wall -pg -O2 -ansi -fPIC -fpermissive -fno-omit-frame-pointer -DNDEBUG -fno-inline-functions -fno-inline-functions-called-once -fno-optimize-sibling-calls
# CFLAGS= -Wall -O3 -fopt-info-optimized
CLIBS= -lm -lpthread
#LAPACK=-L/usr/lib/atlas/sse -llapack -lblas
#LAPACK=-L/usr/local/GotoBLAS2/lib -lgoto2 -lpthread -lgfortran
LAPACK=-L/cm/shared/package/openblas/0.2.17mt/lib -lopenblas -lgfortran -lpthread


INCLUDES= -I. -I../Dirac/ 
LIBPATH=

#### glib
GLIBI=-I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/ -I/usr/lib64/glib-2.0/include
GLIBL=-lglib-2.0

OBJECTS=readsky.o predict.o stationbeam.o predict_withbeam.o transforms.o

default:libsagecal.a
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
libsagecal.a:$(OBJECTS) Radio.h
	ar rv $@ $(OBJECTS); \
	$(RANLIB) $@;
