/* This is a simple program to demostrate the use of libdirac LBFGS routine to solve 
   a general optimization problem.
   This demo tries to solve the problem in stochastic (minibatch) mode.
   Note: The LBFGS routine is fully GPU accelerated,
   so -DHAVE_CUDA=ON should be used when compiling libdirac
*/

#define HAVE_CUDA
/* include Dirac header */
#include <Dirac.h>

#define CUDA_DEBUG
static void
checkCudaError(cudaError_t err, char *file, int line)
{
#ifdef CUDA_DEBUG
    if(!err)
        return;
    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
    exit(EXIT_FAILURE);
#endif
}


/* data structure used by the user specified 
   cost/gradient functions */
typedef struct rosenbrok_data_t_ {
 double alpha;
 /* Unlike in the fullbatch mode, we need to keep a 
   link to a persistent data struct in minibatch mode */
 persistent_data_t *ptdata;
} rosenbrok_data_t;

/* user specified cost function */
/* Note: p is assumed to reside on the device */
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

 cudaError_t err;
 double *phost;
 /* copy 'p' to host */
 err=cudaHostAlloc((void **)&phost, sizeof(double)*m,cudaHostAllocDefault);
 checkCudaError(err,__FILE__,__LINE__);
 err=cudaMemcpy(phost, p, sizeof(double)*m, cudaMemcpyDeviceToHost);
 checkCudaError(err,__FILE__,__LINE__);

 double alpha=t->alpha;
 int ci;
 /* note that we take summation over a minibatch */
 for (ci=ptd->offset; ci<ptd->offset+ptd->nlen; ci++) {
  f=f+alpha*(phost[2*ci+1]-phost[2*ci]*phost[2*ci])*(phost[2*ci+1]-phost[2*ci]*phost[2*ci])+(1.0-phost[2*ci])*(1.0-phost[2*ci]);
 }

 err=cudaFreeHost(phost);
 checkCudaError(err,__FILE__,__LINE__);
 return f;
}


/* user specified gradient function */
/* Note: p,g are assumed to reside on the device */
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

 cudaError_t err;
 double *phost,*ghost;
 /* copy 'p' to host */
 err=cudaHostAlloc((void **)&phost, sizeof(double)*m,cudaHostAllocDefault);
 checkCudaError(err,__FILE__,__LINE__);
 err=cudaMemcpy(phost, p, sizeof(double)*m, cudaMemcpyDeviceToHost);
 checkCudaError(err,__FILE__,__LINE__);

 /* note that we find gradient over a minibatch of ci */
 /* and the remaining values of the gradient is zero */
 /* so first set it to zero */
 err=cudaHostAlloc((void **)&ghost, sizeof(double)*m,cudaHostAllocDefault);
 checkCudaError(err,__FILE__,__LINE__);
 err=cudaMemset(ghost,0,m*sizeof(double));
 checkCudaError(err,__FILE__,__LINE__);

 int ci;
 /* now fill out the non-zero values */
 for (ci=ptd->offset; ci<ptd->offset+ptd->nlen; ci++) {
  ghost[2*ci]=-alpha*4.0*phost[2*ci]*(phost[2*ci+1]-phost[2*ci]*phost[2*ci])-2.0*(1.0-phost[2*ci]);
  ghost[2*ci+1]=alpha*2.0*(phost[2*ci+1]-phost[2*ci]*phost[2*ci]);
 }

 err=cudaMemcpy(g, ghost, sizeof(double)*m, cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);

 err=cudaFreeHost(phost);
 checkCudaError(err,__FILE__,__LINE__);
 err=cudaFreeHost(ghost);
 checkCudaError(err,__FILE__,__LINE__);
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


 /* setup a GPU to use */
 cudaError_t err;
 cublasHandle_t cbhandle;
 cusolverDnHandle_t solver_handle;
 attach_gpu_to_thread(0, &cbhandle, &solver_handle);

 double *pdevice;
 err=cudaMalloc((void**)&(pdevice),m*sizeof(double));
 checkCudaError(err,__FILE__,__LINE__);
 err=cudaMemcpy(pdevice, p, m*sizeof(double), cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);

 /* persistent memory needed for minibatch mode */
 persistent_data_t ptdata;
 int M=5; /* LBFGS memory size */
 int Nt=4; /* how many threads */
 /* how many minibatches */
 int Nminibatch=2;
 /* pre-calculate offset,length of each minibatch */
 int *offsets,*lengths;
 if ((offsets=(int*)calloc((size_t)Nminibatch,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((lengths=(int*)calloc((size_t)Nminibatch,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 int ck=0;
 int batchsize=(m/2+Nminibatch-1)/Nminibatch;

 for (ci=0; ci<Nminibatch; ci++) {
   offsets[ci]=ck;
   if (offsets[ci]+batchsize<=m/2) {
    lengths[ci]=batchsize;
   } else {
    lengths[ci]=m/2-offsets[ci];
   }
   ck=ck+lengths[ci];
 }


 /* we split the m/2 summation terms in the cost function into minibatches */
 lbfgs_persist_init(&ptdata,Nminibatch,m,m/2,M,Nt);
 ptdata.cbhandle=&cbhandle;
 ptdata.solver_handle=&solver_handle;
 /* attach persistent data struct to the user data struct */
 rt.ptdata=&ptdata;

 /* how many epochs to iterate on data */
 int Nepoch=5;
 int nepoch,nbatch;
 /* double loop over epoch and minibatches */
 for (nepoch=0; nepoch<Nepoch; nepoch++) {
  for (nbatch=0; nbatch<Nminibatch; nbatch++) {
   /* get the right offset to the data and the length of data */
   ptdata.offset=offsets[nbatch];
   ptdata.nlen=lengths[nbatch];
   printf("mbatch %d off %d size %d\n",nbatch,ptdata.offset,ptdata.nlen);
   /* The number of iterations per minibatch is smaller */
   lbfgs_fit_cuda(rosenbrok,rosenbrok_grad,pdevice,m,50,M,&rt,&ptdata);
  }
 }
 
 err=cudaMemcpy(p, pdevice, m*sizeof(double), cudaMemcpyDeviceToHost);
 checkCudaError(err,__FILE__,__LINE__);
 err=cudaFree(pdevice);
 checkCudaError(err,__FILE__,__LINE__);
 printf("initial value and solution\n");
 for (ci=0; ci<m; ci++) {
  printf("%lf %lf\n",p0[ci],p[ci]);
 }
 
 free(p0);
 free(p);
 free(offsets);
 free(lengths);
 /* free persistent data */
 detach_gpu_from_thread(cbhandle,solver_handle);
 lbfgs_persist_clear(&ptdata);
 return 0;
}
