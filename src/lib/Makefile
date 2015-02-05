CC=gcc
CXX=g++
CFLAGS= -Wall -O3 -g
CLIBS= -lm -lpthread
#LAPACK=-L/usr/lib/atlas/sse -llapack -lblas
#LAPACK=-L/usr/local/GotoBLAS2/lib -lgoto2 -lpthread -lgfortran
LAPACK=-L/usr/local/OpenBLAS/lib/ -lopenblas -lgfortran -lpthread


INCLUDES= -I. 
LIBPATH=

#### glib
GLIBI=-I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/
GLIBL=-lglib-2.0

OBJECTS=readsky.o dataio.o predict.o lmfit_nocuda.o clmfit_nocuda.o lbfgs_nocuda.o myblas.o residual.o robustlm.o updatenu.o robust_lbfgs_nocuda.o rtr_solve.o  rtr_solve_robust.o manifold_average.o consensus_poly.o rtr_solve_robust_admm.o admm_solve.o

default:libsagecal.a
readsky.o:readsky.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
dataio.o:dataio.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
predict.o:predict.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
lmfit_nocuda.o:lmfit_nocuda.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
clmfit_nocuda.o:clmfit_nocuda.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
lbfgs_nocuda.o:lbfgs_nocuda.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
myblas.o:myblas.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
residual.o:residual.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
robustlm.o:robustlm.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
updatenu.o:updatenu.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
robust_lbfgs_nocuda.o:robust_lbfgs_nocuda.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
rtr_solve.o:rtr_solve.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
rtr_solve_robust.o:rtr_solve_robust.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
manifold_average.o:manifold_average.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
consensus_poly.o:consensus_poly.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
rtr_solve_robust_admm.o:rtr_solve_robust_admm.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<
admm_solve.o:admm_solve.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI)  -c $<


RANLIB=ranlib
libsagecal.a:$(OBJECTS) sagecal.h
	ar rv $@ $(OBJECTS); \
	$(RANLIB) $@;
