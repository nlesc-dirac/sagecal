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


INCLUDES= -I. 
LIBPATH=

#### glib
GLIBI=-I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/ -I/usr/lib64/glib-2.0/include
GLIBL=-lglib-2.0

OBJECTS= lmfit_nocuda.o clmfit_nocuda.o lbfgs_nocuda.o myblas.o residual.o robustlm.o updatenu.o robust_lbfgs_nocuda.o rtr_solve.o  rtr_solve_robust.o manifold_average.o consensus_poly.o rtr_solve_robust_admm.o admm_solve.o

default:libdirac.a
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
libdirac.a:$(OBJECTS) Dirac.h
	ar rv $@ $(OBJECTS); \
	$(RANLIB) $@;

clean:
	rm *.o libdirac.a
