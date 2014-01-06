/*
 *
 Copyright (C) 2006-2008 Sarod Yatawatta <sarod@users.sf.net>  
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 $Id$
 */

#include "data.h"
#include <fstream>
#include <vector> 
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include<sagecal.h>

using namespace std;
using namespace Data;

void
print_copyright(void) {
  cout<<"SAGECal 0.2.8 (C) 2011-2014 Sarod Yatawatta"<<endl;
}


void
print_help(void) {
   cout << "Usage:" << endl;
   cout<<"sagecal -d MS -s sky.txt -c cluster.txt"<<endl;
   cout<<"or"<<endl;
   cout<<"sagecal -f MSlist -s sky.txt -c cluster.txt"<<endl;
   cout << "-d MS name" << endl;
   cout << "-f MSlist: text file with MS names" << endl;
   cout << "-s sky.txt: sky model file"<< endl;
   cout << "-c cluster.txt: cluster file"<< endl;
   cout << "-p solutions.txt: if given, save solution in this file"<< endl;
   cout << "-F sky model format: 0: LSM, 1: LSM with 3 order spectra : default "<< Data::format<<endl;
   cout << "-I input column (DATA/CORRECTED_DATA) : default " <<Data::DataField<< endl;
   cout << "-O ouput column (DATA/CORRECTED_DATA) : default " <<Data::OutField<< endl;
   cout << "-e max EM iterations : default " <<Data::max_emiter<< endl;
   cout << "-g max iterations  (within single EM) : default " <<Data::max_iter<< endl;
   cout << "-l max LBFGS iterations : default " <<Data::max_lbfgs<< endl;
   cout << "-m LBFGS memory size : default " <<Data::lbfgs_m<< endl;
   cout << "-n no of worker threads : default "<<Data::Nt << endl;
   cout << "-t tile size : default " <<Data::TileSize<< endl;
   cout << "-a 0,1 : if 1, only simulate, no calibration: default " <<Data::DoSim<< endl;
   cout << "-b 0,1 : if 1, solve for each channel: default " <<Data::doChan<< endl;
   cout << "-x exclude baselines length (lambda) lower than this in calibration : default "<<Data::min_uvcut << endl;
   cout <<endl<<"Advanced options:"<<endl;
   cout << "-k cluster_id : correct residuals with solution of this cluster : default "<<Data::ccid<< endl;
   cout << "-o robust rho, robust matrix inversion during correction: default "<<Data::rho<< endl;
   cout << "-j 0,1,2... 0 : OSaccel, 1 no OSaccel, 2: OSRLM, 3: RLM: default "<<Data::solver_mode<< endl;
   cout << "-L robust nu, lower bound: default "<<Data::nulow<< endl;
   cout << "-H robust nu, upper bound: default "<<Data::nuhigh<< endl;
   cout << "-R randomize iterations: default "<<Data::randomize<< endl;
//   cout <<endl<<"Dangerous options:"<<endl;
//   cout << "-y longest lambda for tapering (only if >0): default "<<Data::max_uvtaper<< endl;
   cout <<"Report bugs to <sarod@users.sf.net>"<<endl;

}

void 
ParseCmdLine(int ac, char **av) {
    print_copyright();
    char c;
    if(ac < 2)
    {
        print_help();
        exit(0);
    }
    while((c=getopt(ac, av, "a:b:c:d:e:f:g:j:k:l:m:n:o:p:s:t:x:y:F:I:O:L:H:R:h"))!= -1)
    {
        switch(c)
        {
            case 'd':
                TableName = optarg;
                break;
            case 'f':
                MSlist=optarg;
                break;
            case 's':
                SkyModel= optarg;
                break;
            case 'c':
                Clusters= optarg;
                break;
            case 'p':
                solfile= optarg;
                break;
            case 'g':
                max_iter= atoi(optarg);
                break;
            case 'a':
                DoSim= atoi(optarg);
                if (DoSim>1) { DoSim=1; }
                break;
            case 'b':
                doChan= atoi(optarg);
                if (doChan>1) { doChan=1; }
                break;
            case 'F':
                format= atoi(optarg);
                if (format>1) { format=1; }
                break;
            case 'e':
                max_emiter= atoi(optarg);
                break;
            case 'l':
                max_lbfgs= atoi(optarg);
                break;
            case 'm':
                lbfgs_m= atoi(optarg);
                break;
            case 'j':
                solver_mode= atoi(optarg);
                break;
            case 't':
                TileSize = atoi(optarg);
                break;
            case 'I': 
                DataField = optarg;
                break;
            case 'O': 
                OutField = optarg;
                break;
            case 'n': 
                Nt= atoi(optarg);
                break;
            case 'k': 
                ccid= atoi(optarg);
                break;
            case 'o': 
                rho= atof(optarg);
                break;
            case 'L': 
                nulow= atof(optarg);
                break;
            case 'H': 
                nuhigh= atof(optarg);
                break;
            case 'R': 
                randomize= atoi(optarg);
                break;
            case 'x': 
                Data::min_uvcut= atof(optarg);
                break;
            case 'y': 
                Data::max_uvtaper= atof(optarg);
                break;
            case 'h': 
                print_help();
                exit(1);
            default:
                print_help();
                exit(1);
        }
    }

    if (TableName) {
     cout<<" MS: "<<TableName<<endl;
    } else if (MSlist) {
     cout<<" MS list: "<<MSlist<<endl;
    } else {
     print_help();
     exit(1);
    }
    cout<<"Selecting baselines > "<<min_uvcut<<" wavelengths."<<endl;
    if (max_uvtaper>0.0) {
     cout<<"Tapering baselines < "<<max_uvtaper<<" wavelengths."<<endl;
    }
    if (!DoSim) {
    cout<<"Using ";
    if (solver_mode==0 || solver_mode==1) {
     cout<<"Gaussian noise model for solver."<<endl;
    } else {
     cout<<"Robust noise model for solver with degrees of freedom ["<<nulow<<","<<nuhigh<<"]."<<endl;
    }
    } else {
     cout<<"Only doing simulation."<<endl;
    }
}


int 
main(int argc, char **argv) {
    ParseCmdLine(argc, argv);

    if (!Data::SkyModel || !Data::Clusters || !(Data::TableName || Data::MSlist)) {
      print_help();
      exit(1);
    }
    Data::IOData iodata;
    iodata.tilesz=Data::TileSize;
    iodata.deltat=1.0;
    vector<string> msnames;
    if (Data::MSlist) {
     Data::readMSlist(Data::MSlist,&msnames);
    }
    if (Data::TableName) {
     Data::readAuxData(Data::TableName,&iodata);
     cout<<"Only one MS"<<endl;
    } else if (Data::MSlist) {
     Data::readAuxDataList(msnames,&iodata);
     cout<<"Total MS "<<msnames.size()<<endl;
    }
    fflush(stdout);
    if (Data::randomize) {
     srand(time(0)); /* use different seed */
    }
    /**********************************************************/
     int M,Mt,ci,cj,ck;  
  /* parameters */
  double *p,*pinit,*pfreq;
  double **pm;
  complex double *coh;
  FILE *sfp=0;

    if (solfile) {
      if ((sfp=fopen(solfile,"w+"))==0) {
       fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
      return 1;
     }
    }


     double mean_nu;
     clus_source_t *carr;
     baseline_t *barr;
     read_sky_cluster(Data::SkyModel,Data::Clusters,&carr,&M,iodata.freq0,iodata.ra0,iodata.dec0,Data::format);
     printf("Got %d clusters\n",M);
     /* array to store baseline->sta1,sta2 map */
     if ((barr=(baseline_t*)calloc((size_t)iodata.Nbase*iodata.tilesz,sizeof(baseline_t)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     generate_baselines(iodata.Nbase,iodata.tilesz,iodata.N,barr,Data::Nt);

     /* calculate actual no of parameters needed,
      this could be > M */
     Mt=0;
     for (ci=0; ci<M; ci++) {
       //printf("cluster %d has %d time chunks\n",carr[ci].id,carr[ci].nchunk);
       Mt+=carr[ci].nchunk;
     }
     printf("Total effective clusters: %d\n",Mt);

  /* parameters 8*N*M ==> 8*N*Mt */
  if ((p=(double*)calloc((size_t)iodata.N*8*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

#ifdef USE_MIC
  /* need for bitwise copyable parameter passing */
  int *mic_pindex,*mic_chunks;
  if ((mic_chunks=(int*)calloc((size_t)M,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((mic_pindex=(int*)calloc((size_t)Mt,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  int cl=0;
#endif

  /* update cluster array with correct pointers to parameters */
  cj=0;
  for (ci=0; ci<M; ci++) {
    if ((carr[ci].p=(int*)calloc((size_t)carr[ci].nchunk,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
#ifdef USE_MIC
    mic_chunks[ci]=carr[ci].nchunk;
#endif
    for (ck=0; ck<carr[ci].nchunk; ck++) {
      carr[ci].p[ck]=cj*8*iodata.N;
#ifdef USE_MIC
      mic_pindex[cl++]=carr[ci].p[ck];
#endif
      cj++;
    }
  }

  /* pointers to parameters */
  if ((pm=(double**)calloc((size_t)Mt,sizeof(double*)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* setup the pointers */
  for (ci=0; ci<Mt; ci++) {
   pm[ci]=&(p[ci*8*iodata.N]);
  }
  /* initilize parameters to [1,0,0,0,0,0,1,0] */
  for (ci=0; ci<Mt; ci++) {
    for (cj=0; cj<iodata.N; cj++) {
      pm[ci][8*cj]=1.0;
      pm[ci][8*cj+6]=1.0;
    }
  }
  /* backup of default initial values */
  if ((pinit=(double*)calloc((size_t)iodata.N*8*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  memcpy(pinit,p,(size_t)iodata.N*8*Mt*sizeof(double));

  /* coherencies */
  if ((coh=(complex double*)calloc((size_t)(M*iodata.Nbase*iodata.tilesz*4),sizeof(complex double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }



    double res_0,res_1,res_00,res_01;   
   /* previous residual */
   double res_prev=CLM_DBL_MAX;
   double res_ratio=5; /* how much can the residual increase before resetting solutions */
   res_0=res_1=res_00=res_01=0.0;

    /**********************************************************/
    Block<int> sort(1);
    sort[0] = MS::TIME; /* note: only sort over TIME for ms iterator to work */
    /* timeinterval in seconds */
    cout<<"For "<<iodata.tilesz<<" samples, solution time interval (s): "<<iodata.deltat*(double)iodata.tilesz<<endl;
    cout<<"Freq: "<<iodata.freq0/1e6<<" MHz, Chan: "<<iodata.Nchan<<" Bandwidth: "<<iodata.deltaf/1e6<<" MHz"<<endl;
    vector<MSIter*> msitr;
    if (Data::TableName) {
      MeasurementSet *ms=new MeasurementSet(Data::TableName,Table::Update); 
      MSIter *mi=new MSIter(*ms,sort,iodata.deltat*(double)iodata.tilesz);
      msitr.push_back(mi);
    } else if (Data::MSlist) {
     for(int cm=0; cm<iodata.Nms; cm++) {
      MeasurementSet *ms=new MeasurementSet(msnames[cm].c_str(),Table::Update); 
      MSIter *mi=new MSIter(*ms,sort,iodata.deltat*(double)iodata.tilesz);
      msitr.push_back(mi);
     }
    }


    time_t start_time, end_time;
    double elapsed_time;

    int tilex=0;
    for(int cm=0; cm<iodata.Nms; cm++) {
      msitr[cm]->origin();
    }
    /* starting iterations doubled */
    int start_iter=1;
    while (msitr[0]->more()) {
      start_time = time(0);
      if (iodata.Nms==1) {
       Data::loadData(msitr[0]->table(),iodata);
      } else { 
       Data::loadDataList(msitr,iodata);
      }

    /**********************************************************/
    /* update baseline flags */
    /* and set x[]=0 for flagged values */
    preset_flags_and_data(iodata.Nbase*iodata.tilesz,iodata.flag,barr,iodata.x,Data::Nt);

    /* rescale u,v,w by 1/c NOT to wavelengths, that is done later in prediction */
    my_dscal(iodata.Nbase*iodata.tilesz,1.0/CONST_C,iodata.u);
    my_dscal(iodata.Nbase*iodata.tilesz,1.0/CONST_C,iodata.v);
    my_dscal(iodata.Nbase*iodata.tilesz,1.0/CONST_C,iodata.w);

#ifdef USE_MIC
  double *mic_u,*mic_v,*mic_w,*mic_x;
  mic_u=iodata.u;
  mic_v=iodata.v;
  mic_w=iodata.w;
  mic_x=iodata.x;
  int mic_Nbase=iodata.Nbase;
  int mic_tilesz=iodata.tilesz;
  int mic_N=iodata.N;
  double mic_freq0=iodata.freq0;
  double mic_deltaf=iodata.deltaf;
  double mic_data_min_uvcut=Data::min_uvcut;
  int mic_data_Nt=Data::Nt;
  int mic_data_max_emiter=Data::max_emiter;
  int mic_data_max_iter=Data::max_iter;
  int mic_data_max_lbfgs=Data::max_lbfgs;
  int mic_data_lbfgs_m=Data::lbfgs_m;
  int mic_data_gpu_threads=Data::gpu_threads;
  int mic_data_linsolv=Data::linsolv;
  int mic_data_solver_mode=Data::solver_mode;
  int mic_data_randomize=Data::randomize;
  double mic_data_nulow=Data::nulow;
  double mic_data_nuhigh=Data::nuhigh;
#endif

    if (!Data::DoSim) {
    /****************** calibration **************************/
    precalculate_coherencies(iodata.u,iodata.v,iodata.w,coh,iodata.N,iodata.Nbase*iodata.tilesz,barr,carr,M,iodata.freq0,iodata.deltaf,Data::min_uvcut,Data::Nt);
    
#ifndef HAVE_CUDA
    if (start_iter) {
#ifdef USE_MIC
    int mic_data_dochan=Data::doChan;
    #pragma offload target(mic) \
     nocopy(mic_u: length(1) alloc_if(1) free_if(0)) \
     nocopy(mic_v: length(1) alloc_if(1) free_if(0)) \
     nocopy(mic_w: length(1) alloc_if(1) free_if(0)) \
     in(mic_x: length(8*mic_Nbase*mic_tilesz)) \
     in(barr: length(mic_Nbase*mic_tilesz)) \
     in(mic_chunks: length(M)) \
     in(mic_pindex: length(Mt)) \
     in(coh: length(4*M*mic_Nbase*mic_tilesz)) \
     inout(p: length(8*mic_N*Mt)) 
     sagefit_visibilities_mic(mic_u,mic_v,mic_w,mic_x,mic_N,mic_Nbase,mic_tilesz,barr,mic_chunks,mic_pindex,coh,M,Mt,mic_freq0,mic_deltaf,p,mic_data_min_uvcut,mic_data_Nt,2*mic_data_max_emiter,mic_data_max_iter,(mic_data_dochan? 0 :mic_data_max_lbfgs),mic_data_lbfgs_m,mic_data_gpu_threads,mic_data_linsolv,mic_data_solver_mode,mic_data_nulow,mic_data_nuhigh,mic_data_randomize,&mean_nu,&res_0,&res_1);
#else /* NOT MIC */
     sagefit_visibilities(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,2*Data::max_emiter,Data::max_iter,(Data::doChan? 0 :Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
#endif /* USE_MIC */
     start_iter=0;
    } else {
#ifdef USE_MIC
    int mic_data_dochan=Data::doChan;
    #pragma offload target(mic) \
     nocopy(mic_u: length(1) alloc_if(1) free_if(0)) \
     nocopy(mic_v: length(1) alloc_if(1) free_if(0)) \
     nocopy(mic_w: length(1) alloc_if(1) free_if(0)) \
     in(mic_x: length(8*mic_Nbase*mic_tilesz)) \
     in(barr: length(mic_Nbase*mic_tilesz)) \
     in(mic_chunks: length(M)) \
     in(mic_pindex: length(Mt)) \
     in(coh: length(4*M*mic_Nbase*mic_tilesz)) \
     inout(p: length(8*mic_N*Mt)) 
     sagefit_visibilities_mic(mic_u,mic_v,mic_w,mic_x,mic_N,mic_Nbase,mic_tilesz,barr,mic_chunks,mic_pindex,coh,M,Mt,mic_freq0,mic_deltaf,p,mic_data_min_uvcut,mic_data_Nt,mic_data_max_emiter,mic_data_max_iter,(mic_data_dochan? 0: mic_data_max_lbfgs),mic_data_lbfgs_m,mic_data_gpu_threads,mic_data_linsolv,mic_data_solver_mode,mic_data_nulow,mic_data_nuhigh,mic_data_randomize,&mean_nu,&res_0,&res_1);
#else /* NOT MIC */
     sagefit_visibilities(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,(Data::doChan? 0: Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
#endif /* USE_MIC */
    }
#endif /* !HAVE_CUDA */
#ifdef HAVE_CUDA
#ifdef ONE_GPU
    if (start_iter) {
     sagefit_visibilities_dual_pt_one_gpu(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,2*Data::max_emiter,Data::max_iter,(Data::doChan? 0: Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
     start_iter=0;
    } else {
     sagefit_visibilities_dual_pt_one_gpu(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,(Data::doChan? 0:Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
    }
#endif /* ONE_GPU */
#ifndef ONE_GPU
    if (start_iter) {
     sagefit_visibilities_dual_pt_flt(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,2*Data::max_emiter,Data::max_iter,(Data::doChan? 0:Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
     start_iter=0;
    } else {
     sagefit_visibilities_dual_pt_flt(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,(Data::doChan? 0:Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
    }
#endif /* !ONE_GPU */
#endif /* HAVE_CUDA */
   /* if multi channel mode, run BFGS for each channel here 
       and then calculate residuals, else just calculate residuals */
      /* parameters 8*N*M ==> 8*N*Mt */
    if (Data::doChan) {
      if ((pfreq=(double*)calloc((size_t)iodata.N*8*Mt,sizeof(double)))==0) {
        fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
        exit(1);
      }
      double *xfreq;
      if ((xfreq=(double*)calloc((size_t)iodata.Nbase*iodata.tilesz*8,sizeof(double)))==0) {
        fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
        exit(1);
      }

      double deltafch=iodata.deltaf/(double)iodata.Nchan;
      for (ci=0; ci<iodata.Nchan; ci++) {
        memcpy(pfreq,p,(size_t)iodata.N*8*Mt*sizeof(double));
        memcpy(xfreq,&iodata.xo[ci*iodata.Nbase*iodata.tilesz*8],(size_t)iodata.Nbase*iodata.tilesz*8*sizeof(double));
        precalculate_coherencies(iodata.u,iodata.v,iodata.w,coh,iodata.N,iodata.Nbase*iodata.tilesz,barr,carr,M,iodata.freqs[ci],deltafch,Data::min_uvcut,Data::Nt);
      /* FIT, and calculate */
#ifndef HAVE_CUDA
#ifdef USE_MIC
        mic_freq0=iodata.freqs[ci];
        mic_deltaf=deltafch;
     #pragma offload target(mic) \
      nocopy(mic_u: length(1) alloc_if(1) free_if(0)) \
      nocopy(mic_v: length(1) alloc_if(1) free_if(0)) \
      nocopy(mic_w: length(1) alloc_if(1) free_if(0)) \
      in(xfreq: length(8*mic_Nbase*mic_tilesz)) \
      in(barr: length(mic_Nbase*mic_tilesz)) \
      in(mic_chunks: length(M)) \
      in(mic_pindex: length(Mt)) \
      in(coh: length(4*M*mic_Nbase*mic_tilesz)) \
      inout(pfreq: length(8*mic_N*Mt)) 
        bfgsfit_visibilities_mic(mic_u,mic_v,mic_w,xfreq,mic_N,mic_Nbase,mic_tilesz,barr,mic_chunks,mic_pindex,coh,M,Mt,mic_freq0,mic_deltaf,pfreq,mic_data_min_uvcut,mic_data_Nt,mic_data_max_lbfgs,mic_data_lbfgs_m,mic_data_gpu_threads,mic_data_solver_mode,mean_nu,&res_00,&res_01);
        mic_freq0=iodata.freq0;
        mic_deltaf=iodata.deltaf;
#else /* NOT MIC */
        bfgsfit_visibilities(iodata.u,iodata.v,iodata.w,xfreq,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freqs[ci],deltafch,pfreq,Data::min_uvcut,Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00,&res_01);
#endif /* USE_MIC */
#endif /* !HAVE_CUDA */
#ifdef HAVE_CUDA
        bfgsfit_visibilities_gpu(iodata.u,iodata.v,iodata.w,xfreq,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freqs[ci],deltafch,pfreq,Data::min_uvcut,Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00,&res_01);
#endif /* HAVE_CUDA */
        calculate_residuals(iodata.u,iodata.v,iodata.w,pfreq,&iodata.xo[ci*iodata.Nbase*iodata.tilesz*8],iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,iodata.freqs[ci],deltafch,Data::Nt,Data::ccid,Data::rho);
      }
      /* use last solution to save as output */
      memcpy(p,pfreq,(size_t)iodata.N*8*Mt*sizeof(double));
      free(pfreq);
      free(xfreq);
    } else {
     calculate_residuals_multifreq(iodata.u,iodata.v,iodata.w,p,iodata.xo,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,iodata.deltaf,Data::Nt,Data::ccid,Data::rho);
    }
    /****************** end calibration **************************/
   } else {
    /************ simulation only mode ***************************/
    predict_visibilities_multifreq(iodata.u,iodata.v,iodata.w,iodata.xo,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,iodata.deltaf,Data::Nt);
   }

   tilex+=iodata.tilesz;
   /* print solutions to file */
   if (solfile) {
    for (cj=0; cj<iodata.N*8; cj++) {
     fprintf(sfp,"%d ",cj);
     for (ci=M-1; ci>=0; ci--) {
       for (ck=0; ck<carr[ci].nchunk; ck++) {
        fprintf(sfp," %lf",p[carr[ci].p[ck]+cj]);
       }
     }
     fprintf(sfp,"\n");
    }
   }

    /**********************************************************/
      /* also write back */
    if (iodata.Nms==1) {
     Data::writeData(msitr[0]->table(),iodata);
    } else {
     Data::writeDataList(msitr,iodata);
    }
    for(int cm=0; cm<iodata.Nms; cm++) {
      (*msitr[cm])++;
    }
   if (!Data::DoSim) {
   /* if residual has increased too much, reset solutions to original
      initial values */
   if (res_1>res_ratio*res_prev) {
     /* reset solutions so next iteration has default initial values */
     memcpy(p,pinit,(size_t)iodata.N*8*Mt*sizeof(double));
   } else if (res_1<res_prev) { /* only store the min value */
    res_prev=res_1;
   }
   }
    end_time = time(0);
    elapsed_time = ((double) (end_time-start_time)) / 60.0;
    cout<<"nu="<<mean_nu<<endl;
cout<<"Timeslot: "<<tilex<<" Residual: initial="<<res_0<<",final="<<res_1<<", Time spent="<<elapsed_time<<" minutes"<<endl;
    }

   Data::freeData(iodata);


#ifdef USE_MIC
   free(mic_pindex);
   free(mic_chunks);
#endif
    /**********************************************************/

  exinfo_gaussian *exg;
  exinfo_disk *exd;
  exinfo_ring *exr;
  exinfo_shapelet *exs;

  for (ci=0; ci<M; ci++) {
    free(carr[ci].ll);
    free(carr[ci].mm);
    free(carr[ci].nn);
    free(carr[ci].sI);
    free(carr[ci].p);
    for (cj=0; cj<carr[ci].N; cj++) {
     /* do a proper typecast before freeing */
     switch (carr[ci].stype[cj]) {
      case STYPE_GAUSSIAN:
        exg=(exinfo_gaussian*)carr[ci].ex[cj];
        if (exg) free(exg);
        break;
      case STYPE_DISK:
        exd=(exinfo_disk*)carr[ci].ex[cj];
        if (exd) free(exd);
        break;
      case STYPE_RING:
        exr=(exinfo_ring*)carr[ci].ex[cj];
        if (exr) free(exr);
        break;
      case STYPE_SHAPELET:
        exs=(exinfo_shapelet*)carr[ci].ex[cj];
        if (exs)  {
          if (exs->modes) {
            free(exs->modes);
          }
          free(exs);
        }
        break;
      default:
        break;
     }
    }
    free(carr[ci].ex);
    free(carr[ci].stype);
    free(carr[ci].sI0);
    free(carr[ci].f0);
    free(carr[ci].spec_idx);
    free(carr[ci].spec_idx1);
    free(carr[ci].spec_idx2);
  }
  free(carr);
  free(barr);
  free(p);
  free(pinit);
  free(pm);
  free(coh);
  if (solfile) {
    fclose(sfp);
  }
  /**********************************************************/

   cout<<"Done."<<endl;    
   return 0;
}
