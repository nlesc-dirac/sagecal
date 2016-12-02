/*
 *
 Copyright (C) 2006-2015 Sarod Yatawatta <sarod@users.sf.net>  
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
#include "proto.h"
#include <fstream>
#include <vector> 
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#include<sagecal.h>
#include <mpi.h>

#ifndef LMCUT
#define LMCUT 40
#endif

using namespace std;
using namespace Data;

int 
sagecal_slave(int argc, char **argv) {
    ParseCmdLine(argc, argv);
    if (!Data::SkyModel || !Data::Clusters || !Data::MSlist) {
      print_help();
      MPI_Finalize();
      exit(1);
    }

    openblas_set_num_threads(1);//Data::Nt;

    Data::IOData iodata;
    Data::LBeam beam;
    iodata.tilesz=Data::TileSize;
    iodata.deltat=1.0;
    /* determine my MPI rank */
    int myrank;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    /**** get MS from master ***************************************/
    MPI_Probe(0, TAG_MSNAME, MPI_COMM_WORLD, &status);
    int count;
    MPI_Get_count(&status, MPI_CHAR, &count);
    char *buf = new char[count];
    MPI_Recv(buf, count, MPI_CHAR, 0, TAG_MSNAME, MPI_COMM_WORLD, &status);

    Data::TableName=buf;
    cout<<"MS Name "<<Data::TableName<<endl;
    if (Data::TableName) {
     if (!doBeam) {
      Data::readAuxData(Data::TableName,&iodata);
     } else {
      Data::readAuxData(Data::TableName,&iodata,&beam);
     }
    }
    fflush(stdout);

    if (Data::randomize) {
     srand(time(0)); /* use different seed */
    }
    /**********************************************************/
     int M,Mt,ci,cj,ck;  
    /* parameters */
    double *p,*pinit;
    double **pm;
    complex double *coh;
    FILE *sfp=0;
    /* always create default solution file name MS+'.solutions' */
    std::string filebuff=std::string(Data::TableName)+std::string(".solutions\0");
    if ((sfp=fopen(filebuff.c_str(),"w+"))==0) {
       fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
       return 1;
    }
    /* set solfile to non null value */
    solfile=const_cast<char*>(filebuff.c_str());


     double mean_nu;
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
   double res_ratio=15.0; /* how much can the residual increase before resetting solutions, set higher than stand alone mode */
   res_0=res_1=res_00=res_01=0.0;

    /**********************************************************/
    Block<int> sort(1);
    sort[0] = MS::TIME; /* note: only sort over TIME for ms iterator to work */
    /* timeinterval in seconds */
    cout<<"For "<<iodata.tilesz<<" samples, solution time interval (s): "<<iodata.deltat*(double)iodata.tilesz<<endl;
    cout<<"Freq: "<<iodata.freq0/1e6<<" MHz, Chan: "<<iodata.Nchan<<" Bandwidth: "<<iodata.deltaf/1e6<<" MHz"<<endl;
    vector<MSIter*> msitr;
    vector<MeasurementSet*> msvector;
    if (Data::TableName) {
      MeasurementSet *ms=new MeasurementSet(Data::TableName,Table::Update); 
      MSIter *mi=new MSIter(*ms,sort,iodata.deltat*(double)iodata.tilesz);
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
      fprintf(sfp,"# freq(MHz) bandwidth(MHz) time_interval(min) stations clusters effective_clusters\n");
      fprintf(sfp,"%lf %lf %lf %d %d %d\n",iodata.freq0*1e-6,iodata.deltaf*1e-6,(double)iodata.tilesz*iodata.deltat/60.0,iodata.N,M,Mt);
    }


    /**** send info to master ***************************************/
    /* send freq (freq0), no. stations (N), total timeslots (totalt), no. of clusters (M), true no. of clusters with hybrid (Mt), integration time (deltat), bandwidth (deltaf) */
    int *bufint=new int[5];
    double *bufdouble=new double[1];
    bufint[0]=iodata.N;
    bufint[1]=M;
    bufint[2]=Mt;
    bufint[3]=iodata.tilesz;
    bufint[4]=iodata.totalt;
    bufdouble[0]=iodata.freq0;
    MPI_Send(bufint, 5, MPI_INT, 0,TAG_MSAUX, MPI_COMM_WORLD);
    MPI_Send(bufdouble, 1, MPI_DOUBLE, 0,TAG_MSAUX, MPI_COMM_WORLD);

    delete [] bufint;
    delete [] bufdouble;

    /* ADMM memory */
    double *Z,*Y;
    /* Z: (store B_f Z) 2Nx2 x M */
    if ((Z=(double*)calloc((size_t)iodata.N*8*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* Y, 2Nx2 , M times */
    if ((Y=(double*)calloc((size_t)iodata.N*8*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* primal residual J-BZ */
    double *pres;
    if ((pres=(double*)calloc((size_t)iodata.N*8*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }

    double *arho,*arho0;
    if ((arho=(double*)calloc((size_t)M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if ((arho0=(double*)calloc((size_t)M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }

    /* get regularization factor array */
    MPI_Recv(arho,M,MPI_DOUBLE,0,TAG_RHO,MPI_COMM_WORLD,&status);
    /* keep backup of regularization factor */
    memcpy(arho0,arho,(size_t)M*sizeof(double));

    /* if we have more than 1 channel, or if we whiten data, need to backup raw data */
    double *xbackup=0;
    if (iodata.Nchan>1 || Data::whiten) {
      if ((xbackup=(double*)calloc((size_t)iodata.Nbase*8*iodata.tilesz,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
     }
    }


    int msgcode=0;
    /* starting iteration, inner iterations doubled */
    int start_iter=1;
    int sources_precessed=0;

    double inv_c=1.0/CONST_C;

    while(1) {
     start_time = time(0);
     /* get start/end signal from master */
     MPI_Recv(&msgcode,1,MPI_INT,0,TAG_CTRL,MPI_COMM_WORLD,&status);
     if (msgcode==CTRL_END || !msitr[0]->more()) {
cout<<"Slave "<<myrank<<" quitting"<<endl;
      break;
     }
     /* else, load data, do the necessary preprocessing */
     if (!doBeam) {
      Data::loadData(msitr[0]->table(),iodata,&iodata.fratio);
     } else {
      Data::loadData(msitr[0]->table(),iodata,beam,&iodata.fratio);
     }
     /* downweight factor for regularization, depending on amount of data flagged, 
        0.0 means all data are flagged */
     iodata.fratio=1.0-iodata.fratio;
     if (Data::verbose) {
cout<<myrank<<": downweight ratio ("<<iodata.fratio<<") based on flags."<<endl;
     }
     /* send flag ratio (0 means all flagged) to master */
     MPI_Send(&iodata.fratio, 1, MPI_DOUBLE, 0,TAG_FRATIO, MPI_COMM_WORLD);

     /* reweight regularization factors with weight based on flags */
     memcpy(arho,arho0,(size_t)M*sizeof(double));
     my_dscal(M,iodata.fratio,arho);

     /* rescale u,v,w by 1/c NOT to wavelengths, that is done later in prediction */
     my_dscal(iodata.Nbase*iodata.tilesz,inv_c,iodata.u);
     my_dscal(iodata.Nbase*iodata.tilesz,inv_c,iodata.v);
     my_dscal(iodata.Nbase*iodata.tilesz,inv_c,iodata.w);


     /**********************************************************/
     /* update baseline flags */
     /* and set x[]=0 for flagged values */
     preset_flags_and_data(iodata.Nbase*iodata.tilesz,iodata.flag,barr,iodata.x,Data::Nt);
     /* if data is being whitened, whiten x here before copying */
     if (Data::whiten) {
       whiten_data(iodata.Nbase*iodata.tilesz,iodata.x,iodata.u,iodata.v,iodata.freq0,Data::Nt);
     }
     if (iodata.Nchan>1 || Data::whiten) { /* keep fresh copy of raw data */
       my_dcopy(iodata.Nbase*8*iodata.tilesz, iodata.x, 1, xbackup, 1);
     }

     /* precess source locations (also beam pointing) from J2000 to JAPP if we do any beam predictions,
      using first time slot as epoch */
     if (doBeam && !sources_precessed) {
       precess_source_locations(beam.time_utc[iodata.tilesz/2],carr,M,&beam.p_ra0,&beam.p_dec0,Data::Nt);
       sources_precessed=1;
     }
     if (!doBeam) {
      precalculate_coherencies(iodata.u,iodata.v,iodata.w,coh,iodata.N,iodata.Nbase*iodata.tilesz,barr,carr,M,iodata.freq0,iodata.deltaf,iodata.deltat,iodata.dec0,Data::min_uvcut,Data::max_uvcut,Data::Nt);
     } else {
      precalculate_coherencies_withbeam(iodata.u,iodata.v,iodata.w,coh,iodata.N,iodata.Nbase*iodata.tilesz,barr,carr,M,iodata.freq0,iodata.deltaf,iodata.deltat,iodata.dec0,Data::min_uvcut,Data::max_uvcut,
        beam.p_ra0,beam.p_dec0,iodata.freq0,beam.sx,beam.sy,beam.time_utc,iodata.tilesz,beam.Nelem,beam.xx,beam.yy,beam.zz,Data::Nt);
     }
 
     /******************** ADMM  *******************************/

     for (int admm=0; admm<Nadmm; admm++) {
      /* ADMM 1: minimize cost function */
      if (admm==0) { 
#ifndef HAVE_CUDA
      if (start_iter) {
       sagefit_visibilities(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,(iodata.N<=LMCUT?2*Data::max_emiter:4*Data::max_emiter),Data::max_iter,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,(iodata.N<=LMCUT && Data::solver_mode==SM_RTR_OSLM_LBFGS?SM_OSLM_LBFGS:(iodata.N<=LMCUT && (Data::solver_mode==SM_RTR_OSRLM_RLBFGS||Data::solver_mode==SM_NSD_RLBFGS)?SM_OSLM_OSRLM_RLBFGS:Data::solver_mode)),Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1); 
       start_iter=0;
      } else {
       sagefit_visibilities(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
      }
#endif /* !HAVE_CUDA */
#ifdef HAVE_CUDA
      if (start_iter) {
       sagefit_visibilities_dual_pt_flt(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,(iodata.N<=LMCUT?2*Data::max_emiter:4*Data::max_emiter),Data::max_iter,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,(iodata.N<=LMCUT && Data::solver_mode==SM_RTR_OSLM_LBFGS?SM_OSLM_LBFGS:(iodata.N<=LMCUT && (Data::solver_mode==SM_RTR_OSRLM_RLBFGS||Data::solver_mode==SM_NSD_RLBFGS)?SM_OSLM_OSRLM_RLBFGS:Data::solver_mode)),Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
       start_iter=0;
      } else {
       sagefit_visibilities_dual_pt_flt(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
      }
#endif /* HAVE_CUDA */
       /* remember initial residual */
       if (admm==0) {
        res_00=res_0;
        res_01=res_1;
       }
      } else { /* minimize augmented Lagrangian */
       /* since original data is now residual, get a fresh copy of data */
       if (iodata.Nchan>1 || Data::whiten) {
        my_dcopy(iodata.Nbase*8*iodata.tilesz, xbackup, 1, iodata.x, 1);
       } else {
        /* only 1 channel is assumed */
        my_dcopy(iodata.Nbase*8*iodata.tilesz, iodata.xo, 1, iodata.x, 1);
       }
 
#ifndef HAVE_CUDA
       sagefit_visibilities_admm(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Y,Z,Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,0,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,arho,&mean_nu,&res_0,&res_1);
#endif /* !HAVE_CUDA */
#ifdef HAVE_CUDA
       //sagefit_visibilities_admm(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Y,Z,Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,0,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,arho,&mean_nu,&res_0,&res_1);
       sagefit_visibilities_admm_dual_pt_flt(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Y,Z,Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,0,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,arho,&mean_nu,&res_0,&res_1);
#endif /* HAVE_CUDA */
      }

      /* ADMM 2: send Y_i+rho J_i to master */
      /* calculate Y <= Y + rho J */
      if (admm==0) {
       /* Y is set to 0 : so original is just rho * J*/
       my_dcopy(iodata.N*8*Mt, p, 1, Y, 1);
       /* scale by individual rho for each cluster */
       /* if rho<=0, do nothing */
       ck=0;
       for (ci=0; ci<M; ci++) {
        /* Y will be set to 0 if rho<=0 */
        my_dscal(iodata.N*8*carr[ci].nchunk, arho[ci], &Y[ck]);
        ck+=iodata.N*8*carr[ci].nchunk;
       }
      } else {
       ck=0;
       for (ci=0; ci<M; ci++) {
        if (arho[ci]>0.0) {
         my_daxpy(iodata.N*8*carr[ci].nchunk, &p[ck], arho[ci], &Y[ck]);
        }
        ck+=iodata.N*8*carr[ci].nchunk;
//cout<<"Clus="<<ci<<" Chunk="<<carr[ci].nchunk<<" Rho="<<arho[ci]<<endl;
       }
      }

      /* if most data are flagged, only send the original Y we got at the beginning */
      MPI_Send(Y, iodata.N*8*Mt, MPI_DOUBLE, 0,TAG_YDATA, MPI_COMM_WORLD);
      /* for initial ADMM iteration, get back Y with common unitary ambiguity */
      if (admm==0) {
       MPI_Recv(Y, iodata.N*8*Mt, MPI_DOUBLE, 0,TAG_YDATA, MPI_COMM_WORLD, &status);
      }

      /* ADMM 3: get B_i Z from master */
      MPI_Recv(Z, iodata.N*8*Mt, MPI_DOUBLE, 0,TAG_CONSENSUS, MPI_COMM_WORLD, &status);
     
      /* update Y_i <= Y_i + rho (J_i-B_i Z)
          since we already have Y_i + rho J_i, only need -rho (B_i Z) */
      ck=0;
      for (ci=0; ci<M; ci++) {
        if (arho[ci]>0.0) {
         my_daxpy(iodata.N*8*carr[ci].nchunk, &Z[ck], -arho[ci], &Y[ck]);
        }
        ck+=iodata.N*8*carr[ci].nchunk;
      }

      /* calculate primal residual J-BZ */
      my_dcopy(iodata.N*8*Mt, p, 1, pres, 1);
      my_daxpy(iodata.N*8*Mt, Z, -1.0, pres);
      
      /* primal residual : per one real parameter */ 
      /* to remove a load of network traffic and screen output, disable this info */
      if (Data::verbose) {
       cout<<myrank<< ": ADMM : "<<admm<<" residual: primal="<<my_dnrm2(iodata.N*8*Mt,pres)/sqrt((double)8*iodata.N*Mt)<<", initial="<<res_0<<", final="<<res_1<<endl;
      }
     }
     /******************** END ADMM *******************************/

     /* write residuals to output */
     if (!doBeam) {
      calculate_residuals_multifreq(iodata.u,iodata.v,iodata.w,p,iodata.xo,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,iodata.deltaf,iodata.deltat,iodata.dec0,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
     } else {
      calculate_residuals_multifreq_withbeam(iodata.u,iodata.v,iodata.w,p,iodata.xo,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,iodata.deltaf,iodata.deltat,iodata.dec0,
       beam.p_ra0,beam.p_dec0,iodata.freq0,beam.sx,beam.sy,beam.time_utc,beam.Nelem,beam.xx,beam.yy,beam.zz,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
     }
     tilex+=iodata.tilesz;
     /* print solutions to file */
     if (solfile) {
      for (cj=0; cj<iodata.N*8; cj++) {
       fprintf(sfp,"%d ",cj);
       for (ci=M-1; ci>=0; ci--) {
         for (ck=0; ck<carr[ci].nchunk; ck++) {
          /* print solution */
          fprintf(sfp," %e",p[carr[ci].p[ck]+cj]);
         }
       }
       fprintf(sfp,"\n");
      }
     }
     Data::writeData(msitr[0]->table(),iodata);

     /* advance to next data chunk */    
     for(int cm=0; cm<iodata.Nms; cm++) {
       (*msitr[cm])++;
     }
     /* do some quality control */
    /* if residual has increased too much, or all are flagged (0 residual)
      or NaN
      reset solutions to original
      initial values : use residual at 1st ADMM */
    /* do not reset if initial residual is 0, because by def final one will be higher */
    if (res_00!=0.0 && (res_01==0.0 || !isfinite(res_01) || res_01>res_ratio*res_prev)) {
      cout<<"Resetting Solution"<<endl;
      /* reset solutions so next iteration has default initial values */
      memcpy(p,pinit,(size_t)iodata.N*8*Mt*sizeof(double));
      /* also assume iterations have restarted from scratch */
      start_iter=1;
      /* also forget min residual (otherwise will try to reset it always) */
      if (res_01!=0.0 && isfinite(res_01)) {
       res_prev=res_01;
      }
    } else if (res_01<res_prev) { /* only store the min value */
     res_prev=res_01;
    }
    end_time = time(0);
    elapsed_time = ((double) (end_time-start_time)) / 60.0;
    if (solver_mode==SM_OSLM_OSRLM_RLBFGS||solver_mode==SM_RLM_RLBFGS||solver_mode==SM_RTR_OSRLM_RLBFGS || solver_mode==SM_NSD_RLBFGS) { 
     if (Data::verbose) {
      cout<<"nu="<<mean_nu<<endl;
     }
    }
    if (Data::verbose) {
     cout<<myrank<< ": Timeslot: "<<tilex<<" residual: initial="<<res_00<<"/"<<res_0<<",final="<<res_01<<"/"<<res_1<<", Time spent="<<elapsed_time<<" minutes"<<endl;
    }

     /* now send to master signal that we are ready for next data chunk */
     if (start_iter==1) {
      msgcode=CTRL_RESET;
     } else {
      msgcode=CTRL_DONE;
     }

     MPI_Send(&msgcode, 1, MPI_INT, 0,TAG_CTRL, MPI_COMM_WORLD);

    }

    for(int cm=0; cm<iodata.Nms; cm++) {
     delete msitr[cm];
     delete msvector[cm];
    }
    /* free data memory */
    if (!doBeam) {
     Data::freeData(iodata);
    } else {
     Data::freeData(iodata,beam);
    }
 

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
    free(carr[ci].ra);
    free(carr[ci].dec);
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
  if (iodata.Nchan>1 || Data::whiten) {
    free(xbackup);
  }
  free(Z);
  free(Y);
  free(pres);
  free(arho);
  free(arho0);
  /**********************************************************/

   cout<<"Done."<<endl;    
   return 0;
}
