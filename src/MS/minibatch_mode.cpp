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
#include <casacore/casa/Quanta/Quantum.h>

#include "Radio.h"
#include "Dirac.h"
#include "sagecalmain.h"

using namespace std;
using namespace Data;


/* data: with multiple channels, 
   Assume data is very very large
   minibatch: subset of time slots, 
   solutions obtained for every channel - Bandpass calibration
   only using LBFGS (robust): no other solver!
   Future work : add ADMM outer loop
   Future work : add RFI removal from residual
*/


int
run_minibatch_calibration(void) {
    if (!Data::SkyModel || !Data::Clusters || !Data::TableName) {
      print_help();
      exit(1);
    }

    /* how many passes over the whole number of timeslots */
    int nepochs=Data::stochastic_calib_epochs;
    /* how many time slots included in a minibatch */
    int minibatches=Data::stochastic_calib_minibatches;
    int time_per_minibatch=(Data::TileSize+minibatches-1)/minibatches;
    cout<<"Stochastic calibration with "<<nepochs<<" epochs (passes) of "<<minibatches<<" minibatches each for each solution interval."<<endl;
    cout<<"Time per minibatch: "<<time_per_minibatch<<endl;


    /* how many solutions over the bandwidth?
       channels divided to get this many solutions */
    int nsolbw=Data::stochastic_calib_bands;
    int *chanstart,*nchan;

    Data::IOData iodata;
    Data::LBeam beam;
    iodata.tilesz=time_per_minibatch;//original was Data::TileSize;
    iodata.deltat=1.0;
    if (Data::TableName) {
     if (!doBeam) {
      Data::readAuxData(Data::TableName,&iodata);
     } else {
      Data::readAuxData(Data::TableName,&iodata,&beam);
     }
     cout<<"Only one MS"<<endl;
    } 
    fflush(stdout);
    if (Data::randomize) {
     srand(time(0)); /* use different seed */
    }

    /* determine how many channels (max) used per each solution */
    if (nsolbw>=iodata.Nchan) {nsolbw=iodata.Nchan;}
    int nchanpersol=(iodata.Nchan+nsolbw-1)/nsolbw;
    cout<<"Finding "<<nsolbw<<" solutions, each "<<nchanpersol<<" channels wide"<<endl;
    /* allocate memory for solution per channels, 
       channel offset and how many channels */
    if ((chanstart=(int*)calloc((size_t)nsolbw,sizeof(int)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }
    if ((nchan=(int*)calloc((size_t)nsolbw,sizeof(int)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }
    /* determine offset and how many channels for each solution */
    int count=0,ii;
    for (ii=0; ii<nsolbw; ii++) {
      if (count+nchanpersol<iodata.Nchan) {
        nchan[ii]=nchanpersol;
      } else {
        nchan[ii]=iodata.Nchan-count;
      }
      chanstart[ii]=count;

      count+=nchan[ii];
    }

    openblas_set_num_threads(1);//Data::Nt;
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


     /* robust nu is taken from -L option */
     double mean_nu=Data::nulow;
     clus_source_t *carr;
     baseline_t *barr;
     read_sky_cluster(Data::SkyModel,Data::Clusters,&carr,&M,iodata.freq0,iodata.ra0,iodata.dec0,Data::format);
     /* exit if there are 0 clusters (incorrect sky model/ cluster file)*/
     if (M<=0) {
      fprintf(stderr,"%s: %d: no clusters to solve\n",__FILE__,__LINE__);
      exit(1);
     } else {
      printf("Got %d clusters\n",M);
     }

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
  /* parameters for each subband */
  if ((pfreq=(double*)calloc((size_t)iodata.N*8*Mt*nsolbw,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  /* update cluster array with correct pointers to parameters */
  cj=0;
  for (ci=0; ci<M; ci++) {
    if ((carr[ci].p=(int*)calloc((size_t)carr[ci].nchunk,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    for (ck=0; ck<carr[ci].nchunk; ck++) {
      carr[ci].p[ck]=cj*8*iodata.N;
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
  /* initilize parameters to [1,0,0,0,0,0,1,0], or if initsolfile given
     by reading in solutions from that file */
  if (initsolfile) {
      FILE *sfq;
      if ((sfq=fopen(initsolfile,"r"))==0) {
       fprintf(stderr,"%s: %d: no solution file present\n",__FILE__,__LINE__);
       return 1;
      }
      /* remember to skip first 3 lines from solution file */
      char chr;
      for (ci=0; ci<3; ci++) {
       do {
        chr = fgetc(sfq);
       } while (chr != '\n');
      }
     printf("Initializing solutions from %s\n",initsolfile);
     read_solutions(sfq,p,carr,iodata.N,M);
     fclose(sfq);
  } else {
   for (ci=0; ci<Mt; ci++) {
    for (cj=0; cj<iodata.N; cj++) {
      pm[ci][8*cj]=1.0;
      pm[ci][8*cj+6]=1.0;
    }
   }
  }
  free(pm);
  /* backup of default initial values */
  if ((pinit=(double*)calloc((size_t)iodata.N*8*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  memcpy(pinit,p,(size_t)iodata.N*8*Mt*sizeof(double));
  /* now replicate solutions for all subbands */
  for (ii=0; ii<nsolbw; ii++) {
   memcpy(&pfreq[iodata.N*8*Mt*ii],p,(size_t)iodata.N*8*Mt*sizeof(double));
  }

  /* coherencies: note this is only the size of minibatch x number of channels */
  if ((coh=(complex double*)calloc((size_t)(M*iodata.Nbase*iodata.tilesz*4*iodata.Nchan),sizeof(complex double)))==0) {
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
    cout<<"For "<< Data::TileSize<<" samples, solution time interval (s): "<<iodata.deltat*(double)Data::TileSize<<", minibatch (length "<<iodata.tilesz<<" samples) time interval (s): "<< iodata.deltat*(double)iodata.tilesz<<endl;
    cout<<"Freq: "<<iodata.freq0/1e6<<" MHz, Chan: "<<iodata.Nchan<<" Bandwidth: "<<iodata.deltaf/1e6<<" MHz"<<endl;
    /* bandwidth per channel */
    double deltafch=iodata.deltaf/(double)iodata.Nchan;
    vector<MSIter*> msitr;
    vector<MeasurementSet*> msvector;
    if (Data::TableName) {
      MeasurementSet *ms=new MeasurementSet(Data::TableName,Table::Update);
      MSIter *mi=new MSIter(*ms,sort,iodata.deltat*(double)Data::TileSize);
      msitr.push_back(mi);
      msvector.push_back(ms);
    } 

    time_t start_time, end_time;
    double elapsed_time;

    int tilex=0;
    for(int cm=0; cm<iodata.Nms; cm++) {
      msitr[cm]->origin();
    }

    /* write additional info to solution file */
    if (solfile) {
      fprintf(sfp,"# solution file created by SAGECal\n");
      if (nsolbw>1) {
       fprintf(sfp,"# freq(MHz) bandwidth(MHz) channels mini-bands time_interval(min) stations clusters effective_clusters\n");
       fprintf(sfp,"%lf %lf %d %d %lf %d %d %d\n",iodata.freq0*1e-6,iodata.deltaf*1e-6,iodata.Nchan,nsolbw,(double)Data::TileSize*iodata.deltat/60.0,iodata.N,M,Mt);
      } else {
       fprintf(sfp,"# freq(MHz) bandwidth(MHz) time_interval(min) stations clusters effective_clusters\n");
       fprintf(sfp,"%lf %lf %lf %d %d %d\n",iodata.freq0*1e-6,iodata.deltaf*1e-6,(double)Data::TileSize*iodata.deltat/60.0,iodata.N,M,Mt);
      } 
    }

    /* vector for keeping residual of each miniband */
    double *resband;
    if ((resband=(double*)calloc((size_t)nsolbw,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }

    /* starting iterations are doubled */
    int start_iter=1;
    int sources_precessed=0;

    double inv_c=1.0/CONST_C;

#ifdef HAVE_CUDA
   /* setup Heap of GPU,  only need to be done once, before any kernel is launched  */
    if (GPUpredict>0) {
     for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
        cudaSetDevice(gpuid);
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
     }
    }
    /* also attach to a GPU */
    taskhist thst;
    cublasHandle_t cbhandle;
    cusolverDnHandle_t solver_handle;
    init_task_hist(&thst);
    attach_gpu_to_thread(select_work_gpu(MAX_GPU_ID,&thst), &cbhandle, &solver_handle);

    short *hbb;
    int *ptoclus;
    int Nbase1=iodata.Nbase*iodata.tilesz;

    /* auxilliary arrays for GPU */
    if ((hbb=(short*)calloc((size_t)(Nbase1*2),sizeof(short)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* baseline->station mapping */
    rearrange_baselines(Nbase1, barr, hbb, Nt);

    /* parameter->cluster mapping */
    /* for each cluster: chunk size, start param index */
    if ((ptoclus=(int*)calloc((size_t)(2*M),sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    for(ci=0; ci<M; ci++) {
     ptoclus[2*ci]=carr[ci].nchunk;
     ptoclus[2*ci+1]=carr[ci].p[0]; /* so end at p[0]+nchunk*8*N-1 */
    }
#endif
      /* setup persistant struct for the stochastic mode solver */
      /* this will store LBFGS memory and var(grad) parameters */
      /* persistent memory between batches (y,s) pairs
       and info about online var(||grad||) estimate */
      persistent_data_t *ptdata_array;
      if ((ptdata_array=(persistent_data_t*)calloc((size_t)nsolbw,sizeof(persistent_data_t)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
      }
      for (ii=0; ii<nsolbw; ii++) {
        lbfgs_persist_init(&ptdata_array[ii],minibatches,iodata.N*8*Mt,iodata.Nbase*iodata.tilesz,Data::lbfgs_m,Data::gpu_threads);
#ifdef HAVE_CUDA
        /* pointers to cublas/solver */
        ptdata_array[ii].cbhandle=&cbhandle;
        ptdata_array[ii].solver_handle=&solver_handle;
#endif
      }

/******************************* data loop *****************************/
    while (msitr[0]->more()) {
      start_time = time(0);

      res_0=res_1=res_00=res_01=0.0;
      for (int nepch=0; nepch<nepochs; nepch++) {
      for (int nmb=0; nmb<minibatches; nmb++) {

/******************************* work on minibatch *****************************/
        if (!doBeam) {
         Data::loadDataMinibatch(msitr[0]->table(),iodata,nmb,&iodata.fratio);
        } else {
         Data::loadDataMinibatch(msitr[0]->table(),iodata,beam,nmb,&iodata.fratio);
        }
        /* rescale u,v,w by 1/c NOT to wavelengths, that is done later in prediction */
        my_dscal(iodata.Nbase*iodata.tilesz,inv_c,iodata.u);
        my_dscal(iodata.Nbase*iodata.tilesz,inv_c,iodata.v);
        my_dscal(iodata.Nbase*iodata.tilesz,inv_c,iodata.w);

        /**********************************************************/
        /* FIXME: do this efficiently
          update baseline flags */
        /* and set x[]=0 for flagged values */
        preset_flags_and_data(iodata.Nbase*iodata.tilesz,iodata.flag,barr,iodata.x,Data::Nt);
#ifdef HAVE_CUDA
        /* update baseline flags */
        rearrange_baselines(iodata.Nbase*iodata.tilesz, barr, hbb, Nt);
#endif

        /* precess source locations (also beam pointing) from J2000 to JAPP if we do any beam predictions,
           using first time slot as epoch */
        if (doBeam && !sources_precessed) {
         precess_source_locations(beam.time_utc[iodata.tilesz/2],carr,M,&beam.p_ra0,&beam.p_dec0,Data::Nt);
         sources_precessed=1;
        }



    /****************** calibration **************************/
    /* coherency calculation need to be done per channel */
#ifndef HAVE_CUDA
    if (!doBeam) {
     precalculate_coherencies_multifreq(iodata.u,iodata.v,iodata.w,coh,iodata.N,iodata.Nbase*iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,iodata.deltaf,iodata.deltat,iodata.dec0,Data::min_uvcut,Data::max_uvcut,Data::Nt);
    } else {
     //precalculate_coherencies_multifreq_withbeam(iodata.u,iodata.v,iodata.w,coh,iodata.N,iodata.Nbase*iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,iodata.deltaf,iodata.deltat,iodata.dec0,Data::min_uvcut,Data::max_uvcut,
  //beam.p_ra0,beam.p_dec0,iodata.freq0,beam.sx,beam.sy,beam.time_utc,iodata.tilesz,beam.Nelem,beam.xx,beam.yy,beam.zz,Data::Nt);
    }
#endif
#ifdef HAVE_CUDA
   if (GPUpredict) {
     /* note we need to use bandwith per channel here */
     precalculate_coherencies_multifreq_withbeam_gpu(iodata.u,iodata.v,iodata.w,coh,iodata.N,iodata.Nbase*iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,deltafch,iodata.deltat,iodata.dec0,Data::min_uvcut,Data::max_uvcut,
  beam.p_ra0,beam.p_dec0,iodata.freq0,beam.sx,beam.sy,beam.time_utc,iodata.tilesz,beam.Nelem,beam.xx,beam.yy,beam.zz,doBeam,Data::Nt);
   } else {
    if (!doBeam) {
     precalculate_coherencies_multifreq(iodata.u,iodata.v,iodata.w,coh,iodata.N,iodata.Nbase*iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,iodata.deltaf,iodata.deltat,iodata.dec0,Data::min_uvcut,Data::max_uvcut,Data::Nt);
    } else {
     //precalculate_coherencies_withbeam(iodata.u,iodata.v,iodata.w,coh,iodata.N,iodata.Nbase*iodata.tilesz,barr,carr,M,iodata.freq0,iodata.deltaf,iodata.deltat,iodata.dec0,Data::min_uvcut,Data::max_uvcut,
  //beam.p_ra0,beam.p_dec0,iodata.freq0,beam.sx,beam.sy,beam.time_utc,iodata.tilesz,beam.Nelem,beam.xx,beam.yy,beam.zz,Data::Nt);
    }
   }
#endif
     
        /* iterate over solutions covering full bandwidth */
        /* updated values for xo, coh, freqs, Nchan, deltaf needed */
        /*  call LBFGS routine */
      for (ii=0; ii<nsolbw; ii++) {
#ifdef HAVE_CUDA
       bfgsfit_minibatch_visibilities(iodata.u,iodata.v,iodata.w,&iodata.xo[iodata.Nbase*iodata.tilesz*8*chanstart[ii]],iodata.N,iodata.Nbase,iodata.tilesz,hbb,ptoclus,&coh[M*iodata.Nbase*iodata.tilesz*4*chanstart[ii]],M,Mt,&iodata.freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],&pfreq[iodata.N*8*Mt*ii],Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00,&res_01,&ptdata_array[ii],nmb,minibatches);
#else
        bfgsfit_minibatch_visibilities(iodata.u,iodata.v,iodata.w,&iodata.xo[iodata.Nbase*iodata.tilesz*8*chanstart[ii]],iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,&coh[M*iodata.Nbase*iodata.tilesz*4*chanstart[ii]],M,Mt,&iodata.freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],&pfreq[iodata.N*8*Mt*ii],Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00,&res_01,&ptdata_array[ii],nmb,minibatches);
#endif
       res_0+=res_00;
       res_1+=res_01;
       resband[ii]=res_01;
       printf("epoch=%d minibatch=%d band=%d %lf %lf\n",nepch,nmb,ii,res_00,res_01);
      }
    /****************** end calibration **************************/
      /* find average residual over bands*/
      res_0/=(double)nsolbw;
      res_1/=(double)nsolbw;


/******************************* work on minibatch*****************************/
      }
      }


   

      if (start_iter) { start_iter=0; }



    /**********************************************************/
    /* also write back in minibatches */
    for (int nmb=0; nmb<minibatches; nmb++) {
     /* need to load the same data before calculating the residual */
     if (!doBeam) {
        Data::loadDataMinibatch(msitr[0]->table(),iodata,nmb,&iodata.fratio);
     } else {
        Data::loadDataMinibatch(msitr[0]->table(),iodata,beam,nmb,&iodata.fratio);
     }
     /* calculate residual */
        /* rescale u,v,w by 1/c NOT to wavelengths, that is done later in prediction */
        my_dscal(iodata.Nbase*iodata.tilesz,inv_c,iodata.u);
        my_dscal(iodata.Nbase*iodata.tilesz,inv_c,iodata.v);
        my_dscal(iodata.Nbase*iodata.tilesz,inv_c,iodata.w);


#ifndef HAVE_CUDA
      for (ii=0; ii<nsolbw; ii++) {
     if (!doBeam) {
       calculate_residuals_multifreq(iodata.u,iodata.v,iodata.w,&pfreq[iodata.N*8*Mt*ii],&iodata.xo[iodata.Nbase*iodata.tilesz*8*chanstart[ii]],iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,&iodata.freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],iodata.deltat,iodata.dec0,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
     } else {
      calculate_residuals_multifreq_withbeam(iodata.u,iodata.v,iodata.w,&pfreq[iodata.N*8*Mt*ii],&iodata.xo[iodata.Nbase*iodata.tilesz*8*chanstart[ii]],iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,&iodata.freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],iodata.deltat,iodata.dec0,
beam.p_ra0,beam.p_dec0,iodata.freq0,beam.sx,beam.sy,beam.time_utc,beam.Nelem,beam.xx,beam.yy,beam.zz,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
     }
      }
#endif
#ifdef HAVE_CUDA
      for (ii=0; ii<nsolbw; ii++) {
    if (GPUpredict) {
      calculate_residuals_multifreq_withbeam_gpu(iodata.u,iodata.v,iodata.w,&pfreq[iodata.N*8*Mt*ii],&iodata.xo[iodata.Nbase*iodata.tilesz*8*chanstart[ii]],iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,&iodata.freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],iodata.deltat,iodata.dec0,
beam.p_ra0,beam.p_dec0,iodata.freq0,beam.sx,beam.sy,beam.time_utc,beam.Nelem,beam.xx,beam.yy,beam.zz,doBeam,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
    } else {
     if (!doBeam) {
       calculate_residuals_multifreq(iodata.u,iodata.v,iodata.w,&pfreq[iodata.N*8*Mt*ii],&iodata.xo[iodata.Nbase*iodata.tilesz*8*chanstart[ii]],iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,&iodata.freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],iodata.deltat,iodata.dec0,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
     } else {
      calculate_residuals_multifreq_withbeam(iodata.u,iodata.v,iodata.w,&pfreq[iodata.N*8*Mt*ii],&iodata.xo[iodata.Nbase*iodata.tilesz*8*chanstart[ii]],iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,&iodata.freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],iodata.deltat,iodata.dec0,
beam.p_ra0,beam.p_dec0,iodata.freq0,beam.sx,beam.sy,beam.time_utc,beam.Nelem,beam.xx,beam.yy,beam.zz,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
     }
    }
    }
#endif

     Data::writeDataMinibatch(msitr[0]->table(),iodata,nmb);
    }
    for(int cm=0; cm<iodata.Nms; cm++) {
      (*msitr[cm])++;
    }


   tilex+=Data::TileSize;
   /* print solutions to file : columns repeat for each subband */
   if (solfile) {
    for (cj=0; cj<iodata.N*8; cj++) {
     fprintf(sfp,"%d ",cj);
     for (ii=0; ii<nsolbw; ii++) {
     for (ci=M-1; ci>=0; ci--) {
        for (ck=0; ck<carr[ci].nchunk; ck++) {
         fprintf(sfp," %e",pfreq[iodata.N*8*Mt*ii+carr[ci].p[ck]+cj]);
        }
       }
     }
     fprintf(sfp,"\n");
    }
    
   }

   /* Reset solutions in mini-bands with bad residuals */
   for (ii=0; ii<nsolbw; ii++) {
       if (resband[ii]>res_ratio*res_1) {
        cout<<"Resetting solution for band "<<ii<<endl;
        memcpy(&pfreq[iodata.N*8*Mt*ii],pinit,(size_t)iodata.N*8*Mt*sizeof(double));
        lbfgs_persist_reset(&ptdata_array[ii]);
       }
   }


   /* if residual has increased too much, or all are flagged (0 residual)
      or NaN
      reset solutions to original
      initial values */
   if (res_1==0.0 || !isfinite(res_1) || res_1>res_ratio*res_prev) {
     cout<<"Resetting Solution"<<endl;
     /* reset solutions so next iteration has default initial values */
     for (ii=0; ii<nsolbw; ii++) {
       memcpy(&pfreq[iodata.N*8*Mt*ii],pinit,(size_t)iodata.N*8*Mt*sizeof(double));
     }
     /* also assume iterations have restarted from scratch */
     start_iter=1;
     /* also forget min residual (otherwise will try to reset it always) */
     res_prev=res_1;
   } else if (res_1<res_prev) { /* only store the min value */
    res_prev=res_1;
   }
    end_time = time(0);
    elapsed_time = ((double) (end_time-start_time)) / 60.0;
    cout<<"nu="<<mean_nu<<endl;
      cout<<"Timeslot: "<<tilex<<" Residual: initial="<<res_0<<",final="<<res_1<<", Time spent="<<elapsed_time<<" minutes"<<endl;
#ifdef HAVE_CUDA
   /* if -E uses a large value ~say 100, at each multiple of this, clear GPU memory */
   if (GPUpredict>1 && tilex>0 && !(tilex%GPUpredict)) {
    for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
       cudaSetDevice(gpuid);
       cudaDeviceReset();
       cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
    }
   }
#endif

    }
/******************************* end data loop *****************************/
    /* free persistent memory */
    for (ii=0; ii<nsolbw; ii++) {
       lbfgs_persist_clear(&ptdata_array[ii]);
    }
    free(ptdata_array);


    for(int cm=0; cm<iodata.Nms; cm++) {
     delete msitr[cm];
     delete msvector[cm];
    }

   if (!doBeam) {
    Data::freeData(iodata);
   } else {
    Data::freeData(iodata,beam);
   }

    free(nchan);
    free(chanstart);

#ifdef HAVE_CUDA
   detach_gpu_from_thread(cbhandle,solver_handle);
   destroy_task_hist(&thst);
   free(hbb);
   free(ptoclus);
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
    free(carr[ci].sQ);
    free(carr[ci].sU);
    free(carr[ci].sV);
    free(carr[ci].p);
    free(carr[ci].ra);
    free(carr[ci].dec);
    for (cj=0; cj<carr[ci].N; cj++) {
     /* do a proper typecast before freeing */
     switch (carr[ci].stype[cj]) {
      case STYPE_GAUSSIAN:
        exg=(exinfo_gaussian*)carr[ci].ex[cj];
        if (exg) { free(exg); carr[ci].ex[cj]=0; }
        break;
      case STYPE_DISK:
        exd=(exinfo_disk*)carr[ci].ex[cj];
        if (exd) { free(exd); carr[ci].ex[cj]=0; }
        break;
      case STYPE_RING:
        exr=(exinfo_ring*)carr[ci].ex[cj];
        if (exr) { free(exr); carr[ci].ex[cj]=0; }
        break;
      case STYPE_SHAPELET:
        exs=(exinfo_shapelet*)carr[ci].ex[cj];
        if (exs)  {
          if (exs->modes) {
            free(exs->modes);
          }
          free(exs);
          carr[ci].ex[cj]=0;
        }
        break;
      default:
        break;
     }
    }
    free(carr[ci].ex);
    free(carr[ci].stype);
    free(carr[ci].sI0);
    free(carr[ci].sQ0);
    free(carr[ci].sU0);
    free(carr[ci].sV0);
    free(carr[ci].f0);
    free(carr[ci].spec_idx);
    free(carr[ci].spec_idx1);
    free(carr[ci].spec_idx2);
  }
  free(carr);
  free(barr);
  free(p);
  free(pinit);
  free(pfreq);
  free(resband);
  free(coh);
  if (solfile) {
    fclose(sfp);
  }
  /**********************************************************/

   cout<<"Done."<<endl;
   return 0;
}
