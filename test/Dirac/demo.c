/* This is a simple program to demostrate the use of libdirac LBFGS routine to solve 
   a general optimization problem
*/
/* include Dirac header */
#include <Dirac.h>


/* data structure used by the user specified 
   cost/gradient functions */
typedef struct rosenbrok_data_t_ {
 double alpha;
} rosenbrok_data_t;

/* user specified cost function */
/* extended Rosenbrok cost function 
 input : p (mx1) parameters, m: no. of params, adata: additional data
*/
/* note: m should be even here */
/* f= sum_i=0^{m/2-1} (alpha (x_{2i+1}-x_{2i}^2)^2+(1-x_{2i})^2) 
   minimum at x=[1,1,1,..]
*/
double
rosenbrok(double *p, int m, void *adata) {
 double f=0.0;
 rosenbrok_data_t *t=(rosenbrok_data_t*)adata;
 double alpha=t->alpha;
 int ci;
 for (ci=0; ci<m/2; ci++) {
  f=f+alpha*(p[2*ci+1]-p[2*ci]*p[2*ci])*(p[2*ci+1]-p[2*ci]*p[2*ci])+(1.0-p[2*ci])*(1.0-p[2*ci]);
 }
 return f;
}


/* user specified gradient function */
/* gradient
  g_{2i}: -alpha 4 x_{2i}*(x_{2i+1}-x_{2i}^2)-2(1-x_{2i})
  g_{2i+1}: alpha 2 (x_{2i+1}-x_{2i})

  grad function: return gradient (mx1): input : p (mx1) parameters, g (mx1) gradient, m: no. of params, adata: additional data
*/
/* note: m should be even here */
void
rosenbrok_grad(double *p, double *g, int m, void *adata) {
 rosenbrok_data_t *t=(rosenbrok_data_t*)adata;
 double alpha=t->alpha;
 int ci;
 for (ci=0; ci<m/2; ci++) {
  g[2*ci]=-alpha*4.0*p[2*ci]*(p[2*ci+1]-p[2*ci]*p[2*ci])-2.0*(1.0-p[2*ci]);
  g[2*ci+1]=alpha*2.0*(p[2*ci+1]-p[2*ci]*p[2*ci]);
 }

}


int main() {
 rosenbrok_data_t rt;
 rt.alpha=100.0;
 int m=400; /* has to be even */
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


 /* for full batch mode, last argument is NULL */
 lbfgs_fit(rosenbrok,rosenbrok_grad,p,m,30,5,&rt,NULL);
 
 printf("initial value and solution\n");
 for (ci=0; ci<m; ci++) {
  printf("%lf %lf\n",p0[ci],p[ci]);
 }
 
 free(p0);
 free(p);
 return 0;
}
