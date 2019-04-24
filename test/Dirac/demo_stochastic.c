/* This is a simple program to demostrate the use of libdirac LBFGS routine to solve 
   a general optimization problem.
   This demo tries to solve the problem in stochastic (minibatch) mode.
*/
/* include Dirac header */
#include <Dirac.h>


/* data structure used by the user specified 
   cost/gradient functions */
typedef struct rosenbrok_data_t_ {
 double alpha;
 /* Unlike in the fullbatch mode, we need to keep a 
   link to a persistent data struct in minibatch mode */
 persistent_data_t *ptdata;
} rosenbrok_data_t;

/* user specified cost function */
/* extended Rosenbrok cost function 
 input : p (mx1) parameters, m: no. of params, adata: additional data
*/
/* note: m should be even here */
/* full cost function 
   f= sum_i=0^{m/2-1} (alpha (x_{2i+1}-x_{2i}^2)^2+(1-x_{2i})^2) 
   for minibatch, sum_i is taken over a subset of i in [0,m/2-1]
   minimum at x=[1,1,1,..]
*/
double
rosenbrok(double *p, int m, void *adata) {
 double f=0.0;
 rosenbrok_data_t *t=(rosenbrok_data_t*)adata;
 /* get pointer to the persistent data struct */
 persistent_data_t *ptd=t->ptdata;

 double alpha=t->alpha;
 int ci;
 /* note that we take summation over a minibatch */
 for (ci=ptd->offset; ci<ptd->offset+ptd->nlen; ci++) {
  f=f+alpha*(p[2*ci+1]-p[2*ci]*p[2*ci])*(p[2*ci+1]-p[2*ci]*p[2*ci])+(1.0-p[2*ci])*(1.0-p[2*ci]);
 }
 return f;
}


/* user specified gradient function */
/* gradient
  g_{2i}: -alpha 4 x_{2i}*(x_{2i+1}-x_{2i}^2)-2(1-x_{2i})
  g_{2i+1}: alpha 2 (x_{2i+1}-x_{2i})
  In minibatch mode, i is a subset in [0,m/2-1]
  grad function: return gradient (mx1): input : p (mx1) parameters, g (mx1) gradient, m: no. of params, adata: additional data
*/
/* note: m should be even here */
void
rosenbrok_grad(double *p, double *g, int m, void *adata) {
 rosenbrok_data_t *t=(rosenbrok_data_t*)adata;
 double alpha=t->alpha;
 /* get pointer to the persistent data struct */
 persistent_data_t *ptd=t->ptdata;

 int ci;
 /* note that we find gradient over a minibatch of ci */
 /* and the remaining values of the gradient is zero */
 /* so first set it to zero */
 memset(g,0,sizeof(double)*m); 
 /* now fill out the non-zero values */
 for (ci=ptd->offset; ci<ptd->offset+ptd->nlen; ci++) {
  g[2*ci]=-alpha*4.0*p[2*ci]*(p[2*ci+1]-p[2*ci]*p[2*ci])-2.0*(1.0-p[2*ci]);
  g[2*ci+1]=alpha*2.0*(p[2*ci+1]-p[2*ci]*p[2*ci]);
 }

}


int main() {
 rosenbrok_data_t rt;
 rt.alpha=100.0;
 int m=400; /* has to be even */
 /* in order to use stochastic mode, we split the m/2 terms in the 
    cost function to Nminibatch sums. Therefore, the cost/grad 
    functions need to be modified to use minibatches */
 double *p0,*p;
 if ((p0=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((p=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 int ci;
 for (ci=0; ci<m; ci++) {
   p0[ci]=p[ci]=-1.0;
 }

 /* persistent memory needed for minibatch mode */
 persistent_data_t ptdata;
 int M=5; /* LBFGS memory size */
 int Nt=4; /* how many threads */
 /* how many minibatches */
 int Nminibatch=2;
 /* we split the m/2 summation terms in the cost function into minibatches */
 lbfgs_persist_init(&ptdata,Nminibatch,m,m/2,M,Nt);
 /* attach persistent data struct to the user data struct */
 rt.ptdata=&ptdata;

 /* how many epochs to iterate on data */
 int Nepoch=5;
 int nepoch,nbatch;
 /* double loop over epoch and minibatches */
 for (nepoch=0; nepoch<Nepoch; nepoch++) {
  for (nbatch=0; nbatch<Nminibatch; nbatch++) {
   /* get the right offset to the data and the length of data */
   ptdata.offset=ptdata.offsets[nbatch];
   ptdata.nlen=ptdata.lengths[nbatch];
   printf("mbatch %d off %d size %d\n",nbatch,ptdata.offset,ptdata.nlen);
   /* The number of iterations per minibatch is smaller */
   lbfgs_fit(rosenbrok,rosenbrok_grad,p,m,50,M,&rt,&ptdata);
  }
 }
 
 printf("initial value and solution\n");
 for (ci=0; ci<m; ci++) {
  printf("%lf %lf\n",p0[ci],p[ci]);
 }
 
 free(p0);
 free(p);
 /* free persistent data */
 lbfgs_persist_clear(&ptdata);
 return 0;
}
