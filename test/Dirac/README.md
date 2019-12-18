# Using libdirac in general optimization
This directory includes some simple demo programs illustrating how to use optimization routines in libdirac for solving any general optimization problem. We higlight the use of (stochastic) limited-meory Broyden Fletcher Goldfarb Shanno (LBFGS) algorithm.

  * `demo.c`: full batch LBFGS
  * `demo_stochastic.c`: minibatch (stochastic) LBFGS
  * `demo_stochastic_cuda.c`: minibatch LBFGS with full GPU acceleration

Use `Makefile` to build `demo` and `demo_stochastic`. Use `Makefile.cuda` to build `demo_stochastic_cuda`.

Note that for the CUDA demo, libdirac should be built with CUDA support, by using `-DHAVE_CUDA=ON` cmake option.