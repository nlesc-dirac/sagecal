#! /usr/bin/env python

import math
import numpy as np
import argparse

import matplotlib.pyplot as plt
fig=plt.figure()
ax1=fig.add_subplot(211,projection='3d')
ax2=fig.add_subplot(212,projection='3d')

class BeamGenerator:
    def __init__(self,n0=7,beta=0.5, outfile=None, preamble='', verbose=False):
        # file names for .npy models
        self.theta_file_='theta.npy'
        self.phi_file_='phi.npy'
        self.freq_file_='frequency.npy'
        self.ephi_file_='ephi.npy'
        self.etheta_file_='etheta.npy'

        self.out_file_=outfile
        self.out_fd_=None
        self.pream_=preamble

        self.verbose=verbose

        # decomposition parameters
        self.beta_=beta
        self.n0_=n0
        self.nmodes_=n0*(n0+1)//2

    def load_model(self):
        try:
           self.theta_=np.load(self.theta_file_)
           self.phi_=np.load(self.phi_file_)
           self.freq_=np.load(self.freq_file_)
           self.ephi_=np.load(self.ephi_file_)
           self.etheta_=np.load(self.etheta_file_)
        except:
            print('Error: required model files are missing')
            exit(1)

        assert(self.theta_.ndim==1)
        self.n_theta_=self.theta_.size
        assert(self.phi_.ndim==1)
        self.n_phi_=self.phi_.size
        assert(self.freq_.ndim==1)
        self.n_freq_=self.freq_.size

        assert(self.ephi_.shape==(self.n_freq_,self.n_theta_,self.n_phi_))
        assert(self.etheta_.shape==(self.n_freq_,self.n_theta_,self.n_phi_))

        # find index where theta>100 deg, because beyond this (>90) 
        # we model below horizon, and not needed
        self.horizon_idx=np.where(self.theta_>100)[0][0]

        # convert to radians
        self.theta_ *=math.pi/180.0
        self.phi_ *=math.pi/180.0
        
        # determine total power at zenith (theta=0) for all freq
        self.totalpow=np.zeros(self.n_freq_)
        for fr in range(self.n_freq_):
            self.totalpow[fr]=np.sqrt(np.mean(np.abs(self.ephi_[fr,0,:])**2)+np.mean(np.abs(self.etheta_[fr,0,:])**2))

    def Lg(self,p,q,x):
       # generalized Laguerre polynomial L_p^q(x)
       # p,q: integer, x: float, return: float
       if p==0:
          return 1.0
       if p==1:
          return 1.0-x+q
       # use non-recursive form
       L_p=0.0
       L_p1=1.0-x+q
       L_p2=1.0
       for ci in range(2,p+1):
          p1=1.0/ci
          L_p=(2.0+p1*(q-1.0-x))*L_p1-(1.0+p1*(q-1))*L_p2
          L_p2=L_p1
          L_p1=L_p

       return L_p


    def setup_basis(self):
        # setup basis functions (poalr shapelet) for the given model

        preamble=np.zeros(self.nmodes_)
        idx=0
        for n in range(self.n0_):
           for m in range(-n,n+1,2):
             absm=abs(m)
             preamble[idx]=math.sqrt(math.factorial((n-absm)//2)/math.factorial((n+absm)//2)/math.pi)
             if ((n-absm)//2)%2==1 :
                preamble[idx]=-preamble[idx]

             preamble[idx] = preamble[idx]*math.pow(self.beta_,-1.0-absm)

             idx=idx+1

        self.basis_=np.zeros((self.nmodes_,self.horizon_idx,self.n_phi_),dtype=complex)

        for cr in range(self.horizon_idx):
           rb=math.pow(self.theta_[cr]/self.beta_,2.0)
           ex=math.exp(-0.5*rb)
           for ct in range(self.n_phi_):
              idx=0
              for n in range(self.n0_):
                  for m in range(-n,n+1,2):
                     absm=abs(m)
                     Lg1=self.Lg((n-absm)//2,absm,rb)
                     rm=math.pow(math.pi/4+self.theta_[cr],absm)
                     s=math.sin(-m*self.phi_[ct])
                     c=math.cos(-m*self.phi_[ct])
                     pr=preamble[idx]*rm*Lg1*ex
                     self.basis_[idx,cr,ct]=pr*(c+1j*s)
                     idx=idx+1
         
        # create pseudo inverse
        self.A_=self.basis_.reshape(self.nmodes_,-1).T
        self.B_=np.linalg.pinv(self.A_)

    def show_basis(self):
        X,Y=np.meshgrid(self.theta_[:self.horizon_idx],self.phi_)
        for ci in range(self.nmodes_):
          ax1.clear()
          ax1.plot_surface(X*np.cos(Y),X*np.sin(Y),np.real(self.basis_[ci].T),cmap=plt.cm.YlGnBu_r)
          ax1.title.set_text('Real')
          ax2.clear()
          ax2.plot_surface(X*np.cos(Y),X*np.sin(Y),np.imag(self.basis_[ci].T),cmap=plt.cm.YlGnBu_r)
          ax2.title.set_text('Imag')
          plt.savefig('basis_'+str(ci)+'.png')


    def decompose_model_freq(self,freq,component='theta'):
        # decompose model for this freq

        # use normalized pattern
        if component=='theta':
           e_theta=self.etheta_[freq,:self.horizon_idx,:]/self.totalpow[freq]
           b_etheta=e_theta.reshape(-1)
           x=np.matmul(self.B_,b_etheta)
           err=np.linalg.norm(np.matmul(self.A_,x)-b_etheta)/np.linalg.norm(b_etheta)
        else:
           e_phi=self.ephi_[freq,:self.horizon_idx,:]/self.totalpow[freq]
           b_ephi=e_phi.reshape(-1)
           x=np.matmul(self.B_,b_ephi)
           err=np.linalg.norm(np.matmul(self.A_,x)-b_ephi)/np.linalg.norm(b_ephi)

        return x,err


    def decompose_write_header(self):
        self.out_fd_=open(self.out_file_,"w")
        self.out_fd_.write("""#ifndef ELEMENTCOEFF_ALO_H
#define ELEMENTCOEFF_ALO_H
#ifdef __cplusplus
extern \"C\" {
#endif""")
        self.out_fd_.write("""
//This file is automatically generated\n""")
        self.out_fd_.write("//modes="+str(self.nmodes_)+" beta="+str(self.beta_)+"\n")
        self.out_fd_.write("#define ALO_BEAM_ELEM_MODES "+str(self.n0_)+"\n")
        self.out_fd_.write("#define ALO_BEAM_ELEM_BETA "+str(self.beta_)+"\n")
        self.out_fd_.write("#define ALO_FREQS "+str(self.n_freq_)+"\n")
        self.out_fd_.write("//Frequency coordinate GHz"+"\n")
        self.out_fd_.write("const static double "+str(self.pream_)+"_beam_elem_freqs["+str(self.n_freq_)+"]={\n")
        for freq in range(self.n_freq_):
            self.out_fd_.write(f" {self.freq_[freq]/1e9:10.9f},")
        self.out_fd_.write("};"+"\n")
        self.out_fd_.write("const static complex double "+str(self.pream_)+"_beam_elem_theta["+str(self.n_freq_)+"]["+str(self.nmodes_)+"]={\n")
        for freq in range(self.n_freq_):
            x,err=self.decompose_model_freq(freq,'theta')
            self.out_fd_.write("{"+"\n")
            for nm in range(self.nmodes_):
               self.out_fd_.write(f" {np.real(x[nm]):e}+_Complex_I*({np.imag(x[nm]):e}),")
            self.out_fd_.write("},"+"\n")
            if self.verbose:
               print(f'Freq {freq} theta error {err}')

        self.out_fd_.write("};"+"\n")
        self.out_fd_.write("const static complex double "+str(self.pream_)+"_beam_elem_phi["+str(self.n_freq_)+"]["+str(self.nmodes_)+"]={\n")
        for freq in range(self.n_freq_):
            x,err=self.decompose_model_freq(freq,'phi')
            self.out_fd_.write("{"+"\n")
            for nm in range(self.nmodes_):
               self.out_fd_.write(f" {np.real(x[nm]):e}+_Complex_I*({np.imag(x[nm]):e}),")
            self.out_fd_.write("},"+"\n")
            if self.verbose:
               print(f'Freq {freq} phi error {err}')

        self.out_fd_.write("};"+"\n")

        self.out_fd_.write("""#ifdef __cplusplus
    } /* extern "C" */
#endif
#endif /* ELEMENTCOEFF_ALO_H */""")


    def __del__(self):
        if self.out_fd_:
            self.out_fd_.close()

def main(args):
    bg=BeamGenerator(n0=args.order, beta=args.scale, outfile=args.output, preamble=args.preamble, verbose=args.verbose)
    bg.load_model()
    bg.setup_basis()
    if args.show:
       bg.show_basis()
    bg.decompose_write_header()


if __name__=='__main__':
    parser=argparse.ArgumentParser(
      description='Generate element beam coefficients for sagecal',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--scale',type=float,default=0.5,
        help='basis scale factor')
    parser.add_argument('--order',type=int,default=7,
        help='basis model order')
    parser.add_argument('--output',type=str,default='output.h',
        help='write output to this (C header) file')
    parser.add_argument('--preamble',type=str,default='alo',
        help='string to uniquely indetify this model')
    parser.add_argument('--show', action='store_true', default=False,
       help='show basis functions')
    parser.add_argument('--verbose', action='store_true', default=False,
       help='print reconstruction error')
 
    args=parser.parse_args()
    if args.output:
      main(args)
    else:
      parser.print_help()
