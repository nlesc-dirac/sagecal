/*
 *
 Copyright (C) 2020 Sarod Yatawatta <sarod@users.sf.net>  
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

#include <map>
#include <string>
#include <cstring>
#include <glob.h>


#include <Dirac.h>
#include <Radio.h>
#include <mpi.h>

#ifndef LMCUT
#define LMCUT 40
#endif

using namespace std;
using namespace Data;
//#define DEBUG

int 
sagecal_stochastic_slave(int argc, char **argv) {
    ParseCmdLine(argc, argv);
    if (!Data::SkyModel || !Data::Clusters || !Data::MSpattern) {
      print_help();
      MPI_Finalize();
      exit(1);
    }

    openblas_set_num_threads(1);

    /* determine my MPI rank */
    int myrank;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    /**** setup MS names ***************************************/
    // slave sends host name to master
    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(name, &len);
    MPI_Send(name,strlen(name),MPI_CHAR,0,TAG_MSNAME,MPI_COMM_WORLD);

    //slaves wait for request from master, or idle signal
    int count;
    MPI_Probe(0,TAG_MSNAME,MPI_COMM_WORLD,&status);
    MPI_Get_count(&status,MPI_CHAR,&count);
    // allocate memory
    char *buf=new char[count];
    MPI_Recv(buf,count,MPI_CHAR,0,TAG_MSNAME,MPI_COMM_WORLD,&status);
    string buf1(buf,count);
    delete [] buf;
    //if we have received a non null string
    if (count > 0) {
      // search for matching file names
      glob_t glob_result;
      int globret=glob(buf1.c_str(),GLOB_TILDE,NULL,&glob_result);
      vector<string> ret;
      for(unsigned int i=0;i<glob_result.gl_pathc;++i){
          ret.push_back(string(glob_result.gl_pathv[i]));
      }
      globfree(&glob_result);
      // check for errors and exist
      if (globret>0) {
cout<<"Error in checking files matching pattern "<<buf1<<". Exiting."<<endl;
       exit(1);
      } else {
       //slave finds the file list and send back to master
       int nfiles=ret.size();
        MPI_Send(&nfiles,1,MPI_INT,0,TAG_MSNAME,MPI_COMM_WORLD);
        //also send the file names
        int cj;
        for (cj=0; cj<nfiles; cj++) {
          MPI_Send(ret[cj].c_str(),ret[cj].length(),MPI_CHAR,0,TAG_MSNAME,MPI_COMM_WORLD);
        }
      }
   }

     //get back the MS names for this slave
     vector<string> myms;
     vector<int> myids;
     int mymscount,mymsid;
     MPI_Recv(&mymscount,1,MPI_INT,0,TAG_MSNAME,MPI_COMM_WORLD,&status);
     if (mymscount>0) {
      for (int ch=0; ch<mymscount; ch++) {
        MPI_Probe(0,TAG_MSNAME,MPI_COMM_WORLD,&status);
        MPI_Get_count(&status,MPI_CHAR,&count);
        buf=new char[count];
        MPI_Recv(buf,count,MPI_CHAR,0,TAG_MSNAME,MPI_COMM_WORLD,&status);
        string buf2(buf,count);
        myms.push_back(buf2);
        // also get the id of this MS
        MPI_Recv(&mymsid,1,MPI_INT,0,TAG_MSNAME,MPI_COMM_WORLD,&status);
        myids.push_back(mymsid);

        delete [] buf;
      }
     } else {
cout<<"Slave "<<myrank<<" has nothing to do"<<endl;
       return 0;
     }


    /**** end setup MS names ***************************************/
    /* how many passes over the whole number of timeslots */
    int nepochs=Data::stochastic_calib_epochs;
    /* how many time slots included in a minibatch */
    int minibatches=Data::stochastic_calib_minibatches;
    int time_per_minibatch=(Data::TileSize+minibatches-1)/minibatches;
    cout<<"Stochastic calibration with "<<nepochs<<" epochs (passes) of "<<minibatches<<" minibatches each for each solution interval."<<endl;
    cout<<"Time per minibatch: "<<time_per_minibatch<<endl;
    cout<<"ADMM iterations="<<Nadmm<<" polynomial order="<<Npoly<<" regularization="<<admm_rho<<endl;


    /* how many solutions over the bandwidth?
       channels divided to get this many solutions */
    int nsolbw=Data::stochastic_calib_bands;
    int *chanstart,*nchan;

    //create vectors to store data, beam, sky info etc for each MS
    vector<Data::IOData> iodata_vec(mymscount);
    vector<Data::LBeam> beam_vec(mymscount);

    if (Data::randomize) {
     srand(time(0)); /* use different seed */
    } else {
     srand(0);
    }
    /**********************************************************/
     int M,Mt,ci,cj,ck;  
    /* parameters */
    double *pinit;

    for(int cm=0; cm<mymscount; cm++) {
     iodata_vec[cm].tilesz=time_per_minibatch;//original was Data::TileSize;
     iodata_vec[cm].deltat=1.0;
     if (!doBeam) {
      Data::readAuxData(myms[cm].c_str(),&iodata_vec[cm]);
     } else {
      Data::readAuxData(myms[cm].c_str(),&iodata_vec[cm],&beam_vec[cm]);
     }
    }
    fflush(stdout);

    /* cannot run ADMM if we have only one channel, so print error and exit */
    if (iodata_vec[0].Nchan==1) {
      fprintf(stderr,"Not possible to run consensus optimization with only %d channels.\n Quitting.\n",iodata_vec[0].Nchan);
      exit(1);
    }

    /* determine how many channels (max) used per each solution */
    if (nsolbw>=iodata_vec[0].Nchan) {nsolbw=iodata_vec[0].Nchan;}
    int nchanpersol=(iodata_vec[0].Nchan+nsolbw-1)/nsolbw;
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
    int ii;
    count=0;
    for (ii=0; ii<nsolbw; ii++) {
      if (count+nchanpersol<iodata_vec[0].Nchan) {
        nchan[ii]=nchanpersol;
      } else {
        nchan[ii]=iodata_vec[0].Nchan-count;
      }
      chanstart[ii]=count;

      count+=nchan[ii];
    }

    vector<FILE *> sfp_vec(mymscount);
    for(int cm=0; cm<mymscount; cm++) {
     /* always create default solution file name MS+'.solutions' */
     string filebuff=std::string(myms[cm])+std::string(".solutions\0");
     if ((sfp_vec[cm]=fopen(filebuff.c_str(),"w+"))==0) {
       fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
       exit(1);
     }
    }

     /* robust nu is taken from -L option */
     double mean_nu=Data::nulow;
     vector<clus_source_t *> carr_vec(mymscount);
     vector<baseline_t *> barr_vec(mymscount);

     for(int cm=0; cm<mymscount; cm++) {
      read_sky_cluster(Data::SkyModel,Data::Clusters,&carr_vec[cm],&M,iodata_vec[cm].freq0,iodata_vec[cm].ra0,iodata_vec[cm].dec0,Data::format);
     }

     /* exit if there are 0 clusters (incorrect sky model/ cluster file)*/
     if (M<=0) {
      fprintf(stderr,"%s: %d: no clusters to solve\n",__FILE__,__LINE__);
      exit(1);
     } else {
      printf("Got %d clusters\n",M);
     }

     /* array to store baseline->sta1,sta2 map */
     for(int cm=0; cm<mymscount; cm++) {
      if ((barr_vec[cm]=(baseline_t*)calloc((size_t)iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,sizeof(baseline_t)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      generate_baselines(iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,iodata_vec[cm].N,barr_vec[cm],Data::Nt);
     }

     /* calculate actual no of parameters needed,
      this could be > M */
     Mt=0;
     for (ci=0; ci<M; ci++) {
       //printf("cluster %d has %d time chunks\n",carr_vec[0][ci].id,carr_vec[0][ci].nchunk);
       Mt+=carr_vec[0][ci].nchunk;
     }
     printf("Total effective clusters: %d\n",Mt);

    vector<double *> p_vec(mymscount);
    vector<double *> pfreq_vec(mymscount);
    vector<double **> pm_vec(mymscount);

    for(int cm=0; cm<mymscount; cm++) {
    /* parameters 8*N*M ==> 8*N*Mt */
     if ((p_vec[cm]=(double*)calloc((size_t)iodata_vec[cm].N*8*Mt,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if ((pfreq_vec[cm]=(double*)calloc((size_t)iodata_vec[cm].N*8*Mt*nsolbw,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
    }
    for(int cm=0; cm<mymscount; cm++) {
     /* update cluster array with correct pointers to parameters */
     cj=0;
     for (ci=0; ci<M; ci++) {
       if ((carr_vec[cm][ci].p=(int*)calloc((size_t)carr_vec[cm][ci].nchunk,sizeof(int)))==0) {
        fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
        exit(1);
       }
       for (ck=0; ck<carr_vec[cm][ci].nchunk; ck++) {
         carr_vec[cm][ci].p[ck]=cj*8*iodata_vec[cm].N;
         cj++;
       }
     }
   }

  for(int cm=0; cm<mymscount; cm++) {
   /* pointers to parameters */
   if ((pm_vec[cm]=(double**)calloc((size_t)Mt,sizeof(double*)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
   }
   /* setup the pointers */
   for (ci=0; ci<Mt; ci++) {
    pm_vec[cm][ci]=&(p_vec[cm][ci*8*iodata_vec[cm].N]);
   }
   /* initilize parameters to [1,0,0,0,0,0,1,0] */
   if (!initsolfile) {
    for (ci=0; ci<Mt; ci++) {
     for (cj=0; cj<iodata_vec[cm].N; cj++) {
       pm_vec[cm][ci][8*cj]=1.0;
       pm_vec[cm][ci][8*cj+6]=1.0;
     }
    }
   } 
  }
  /* initialize solutions by reading from a file */
  if ((pinit=(double*)calloc((size_t)iodata_vec[0].N*8*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if (initsolfile) {
      FILE *sfq;
      if ((sfq=fopen(initsolfile,"r"))==0) {
       fprintf(stderr,"%s: %d: no solution file present\n",__FILE__,__LINE__);
       exit(1);
      }
      /* remember to skip first 3 lines from solution file */
      char chr;
      for (ci=0; ci<3; ci++) {
       do {
        chr = fgetc(sfq);
       } while (chr != '\n');
      }
     printf("Initializing solutions from %s\n",initsolfile);
     read_solutions(sfq,pinit,carr_vec[0],iodata_vec[0].N,M);
     fclose(sfq);

   /* backup of default initial values */
   for(int cm=0; cm<mymscount; cm++) {
    memcpy(p_vec[cm],pinit,(size_t)iodata_vec[0].N*8*Mt*sizeof(double));
   }
  } else {
    /* backup of default initial values */
    memcpy(pinit,p_vec[0],(size_t)iodata_vec[0].N*8*Mt*sizeof(double));
  }
  /* now replicate solutions for all subbands */
  for(int cm=0; cm<mymscount; cm++) {
   for (ii=0; ii<nsolbw; ii++) {
    memcpy(&pfreq_vec[cm][iodata_vec[cm].N*8*Mt*ii],p_vec[cm],(size_t)iodata_vec[cm].N*8*Mt*sizeof(double));
   }
  }


  vector<complex double *> coh_vec(mymscount);
  /* coherencies */
  for(int cm=0; cm<mymscount; cm++) {
   if ((coh_vec[cm]=(complex double*)calloc((size_t)(M*iodata_vec[cm].Nbase*iodata_vec[cm].tilesz*4*iodata_vec[0].Nchan),sizeof(complex double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
   }
  }


   double res_0,res_1;
   vector<double> res_00(mymscount),res_01(mymscount),res_prev(mymscount);   
   /* how much can the residual increase before resetting solutions, 
      use a lower value here (original 5) for more robustness, also because this is a log() cost */
   double res_ratio=1.5; /* how much can the residual increase before resetting solutions, set higher than stand alone mode */
    res_0=res_1=0.0;
    for(int cm=0; cm<mymscount; cm++) {
     res_00[cm]=res_01[cm]=0.0;
     /* previous residual */
     res_prev[cm]=CLM_DBL_MAX;
    }

    /**********************************************************/
    Block<int> sort(1);
    sort[0] = MS::TIME; /* note: only sort over TIME for ms iterator to work */
    /* timeinterval in seconds */

    for(int cm=0; cm<mymscount; cm++) {
     cout<<"For "<<Data::TileSize<<" samples, solution time interval (s): "<<iodata_vec[cm].deltat*(double)Data::TileSize<<", minibatch (length "<<iodata_vec[cm].tilesz<<" samples) time interval (s): "<< iodata_vec[cm].deltat*(double)iodata_vec[cm].tilesz<<endl;
     cout<<"Freq: "<<iodata_vec[cm].freq0/1e6<<" MHz, Chan: "<<iodata_vec[cm].Nchan<<" Bandwidth: "<<iodata_vec[cm].deltaf/1e6<<" MHz"<<endl;
    }
    /* bandwidth per channel */
    double deltafch=iodata_vec[0].deltaf/(double)iodata_vec[0].Nchan;

    vector<MSIter*> msitr;
    vector<MeasurementSet*> msvector;
    for(int cm=0; cm<mymscount; cm++) {
      MeasurementSet *ms=new MeasurementSet(myms[cm],Table::Update); 
      MSIter *mi=new MSIter(*ms,sort,iodata_vec[cm].deltat*(double)Data::TileSize);
      msitr.push_back(mi);
      msvector.push_back(ms);
    }
   
    time_t start_time, end_time;
    double elapsed_time;

    int tilex=0;
    for(int cm=0; cm<mymscount; cm++) {
      msitr[cm]->origin();
    }

    /* write additional info to solution file */
    for(int cm=0; cm<mymscount; cm++) {
     fprintf(sfp_vec[cm],"# solution file created by SAGECal\n");
     if (nsolbw>1) {
       fprintf(sfp_vec[cm],"# freq(MHz) bandwidth(MHz) channels mini-bands time_interval(min) stations clusters effective_clusters\n");
       fprintf(sfp_vec[cm],"%lf %lf %d %d %lf %d %d %d\n",iodata_vec[cm].freq0*1e-6,iodata_vec[cm].deltaf*1e-6,iodata_vec[cm].Nchan,nsolbw,(double)Data::TileSize*iodata_vec[cm].deltat/60.0,iodata_vec[cm].N,M,Mt);
     } else {
      fprintf(sfp_vec[cm],"# freq(MHz) bandwidth(MHz) time_interval(min) stations clusters effective_clusters\n");
      fprintf(sfp_vec[cm],"%lf %lf %lf %d %d %d\n",iodata_vec[cm].freq0*1e-6,iodata_vec[cm].deltaf*1e-6,(double)Data::TileSize*iodata_vec[cm].deltat/60.0,iodata_vec[cm].N,M,Mt);
     }
    }



    /**** send info to master ***************************************/
    /* send msid, freq (freq0), no. stations (N), total timeslots (totalt), no. of clusters (M), true no. of clusters with hybrid (Mt), integration time (deltat), bandwidth (deltaf) */
    int *bufint=new int[6];
    double *bufdouble=new double[1];
    for(int cm=0; cm<mymscount; cm++) {
     bufint[0]=myids[cm];
     bufint[1]=iodata_vec[cm].N;
     bufint[2]=M;
     bufint[3]=Mt;
     bufint[4]=Data::TileSize;
     bufint[5]=iodata_vec[cm].totalt;
     bufdouble[0]=iodata_vec[cm].freq0;
     MPI_Send(bufint, 6, MPI_INT, 0,TAG_MSAUX, MPI_COMM_WORLD);
     MPI_Send(bufdouble, 1, MPI_DOUBLE, 0,TAG_MSAUX, MPI_COMM_WORLD);
    }

    delete [] bufint;
    delete [] bufdouble;


    /* ADMM memory */
    double *Z,*Zold,*Zavg,*Y,*z,*B,*Bii,*rhok;
    /* Z: 2Nx2 x Npoly x Mt */
    /* keep ordered by Mt (one direction together) */
    if ((Z=(double*)calloc((size_t)iodata_vec[0].N*8*Npoly*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if ((Zold=(double*)calloc((size_t)iodata_vec[0].N*8*Npoly*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if ((Zavg=(double*)calloc((size_t)iodata_vec[0].N*8*Npoly*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* z : 2Nx2 x Mt x Npoly vector, so each block is 8NMt */
    if ((z=(double*)calloc((size_t)iodata_vec[0].N*8*Npoly*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* copy of Y+rho J, Mt times, for each solution */
    /* keep ordered by M (one direction together) */
    if ((Y=(double*)calloc((size_t)iodata_vec[0].N*8*Mt*nsolbw,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }

    /* Npoly terms, for each solution, so Npoly x nsolbw */
    if ((B=(double*)calloc((size_t)Npoly*nsolbw,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* pseudoinverse  Mt values of NpolyxNpoly matrices */
    if ((Bii=(double*)calloc((size_t)Mt*Npoly*Npoly,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* each Mt block is for one freq */
    if ((rhok=(double*)calloc((size_t)Mt*nsolbw,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }

    /* freq. vector at each solution is taken */
    double *ffreq;
    if ((ffreq=(double*)calloc((size_t)nsolbw,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* also find the min,max freq values where solution is taken */
    double min_f,max_f;
    min_f=1e15; max_f=-1e15;
    for (ii=0; ii<nsolbw; ii++) {
      /* consider the case where frequencies are non regular */
      for (int fii=0; fii<nchan[ii]; fii++) {
        ffreq[ii]+=iodata_vec[0].freqs[chanstart[ii]+fii];
      }
      ffreq[ii]/=(double)(nchan[ii]); /* mean */
      printf("%d %lf %lf\n",ii,ffreq[ii],iodata_vec[0].freq0);
      min_f=(min_f>ffreq[ii]?ffreq[ii]:min_f);
      max_f=(max_f<ffreq[ii]?ffreq[ii]:max_f);
    }

    /* send min,max freq to master */
    bufdouble=new double[2];
    bufdouble[0]=min_f;
    bufdouble[1]=max_f;

    MPI_Send(bufdouble, 2, MPI_DOUBLE, 0,TAG_MSAUX, MPI_COMM_WORLD);

    delete [] bufdouble;
    bufdouble=new double[3];
    /* get back global min,max freq from master */
    MPI_Recv(bufdouble, 3, /* min, max freq */
           MPI_DOUBLE, 0, TAG_MSAUX, MPI_COMM_WORLD, &status);
    min_f=bufdouble[0];
    max_f=bufdouble[1];
    double ref_f=bufdouble[2];
    printf("%d New range %lf %lf %lf\n",myrank,min_f,max_f,ref_f);
    delete [] bufdouble;
    /* resize memory so that polynomials are generated for expanded freq range [min_f,ffreq,max_f] */
    double *Bext, *ffreq2;
    if ((Bext=(double*)calloc((size_t)Npoly*(nsolbw+2),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if ((ffreq2=(double*)calloc((size_t)(nsolbw+2),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    memcpy(ffreq2,ffreq,(size_t)(nsolbw)*sizeof(double));
    ffreq2[nsolbw]=min_f;
    ffreq2[nsolbw+1]=max_f;


    /* setup polynomials */
    setup_polynomials(Bext, Npoly, nsolbw+2, ffreq2, ref_f,(Npoly==1?1:PolyType));

    memcpy(B,Bext,(size_t)(nsolbw*Npoly)*sizeof(double));

    double alpha=Data::federated_reg_alpha;
    setweights(Mt*nsolbw,rhok,Data::admm_rho,Data::Nt);
    /* find inverse of B for each cluster, solution, alpha is fed. avg. regularization */
    find_prod_inverse_full_fed(B,Bii,Npoly,nsolbw,Mt,rhok,alpha,Data::Nt);

    free(ffreq);
    free(ffreq2);
    free(Bext);

    /* vector for keeping residual of each miniband and flag vector */
    double *resband;
    int *fband;
    if ((resband=(double*)calloc((size_t)nsolbw,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if ((fband=(int*)calloc((size_t)nsolbw,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }

    int msgcode=0;
    /* starting iteration, inner iterations doubled */
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
    int Nbase1=iodata_vec[0].Nbase*iodata_vec[0].tilesz;

    /* auxilliary arrays for GPU */
    if ((hbb=(short*)calloc((size_t)(Nbase1*2),sizeof(short)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* baseline->station mapping */
    rearrange_baselines(Nbase1, barr_vec[0], hbb, Nt);

    /* parameter->cluster mapping */
    /* for each cluster: chunk size, start param index */
    if ((ptoclus=(int*)calloc((size_t)(2*M),sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    for(ci=0; ci<M; ci++) {
     ptoclus[2*ci]=carr_vec[0][ci].nchunk;
     ptoclus[2*ci+1]=carr_vec[0][ci].p[0]; /* so end at p[0]+nchunk*8*N-1 */
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
        lbfgs_persist_init(&ptdata_array[ii],minibatches,iodata_vec[0].N*8*Mt,iodata_vec[0].Nbase*iodata_vec[0].tilesz,Data::lbfgs_m,Data::gpu_threads);
#ifdef HAVE_CUDA
        /* pointers to cublas/solver */
        ptdata_array[ii].cbhandle=&cbhandle;
        ptdata_array[ii].solver_handle=&solver_handle;
#endif
      }




    while(1) {
     start_time = time(0);
     /* get start/end signal from master */
     MPI_Recv(&msgcode,1,MPI_INT,0,TAG_CTRL,MPI_COMM_WORLD,&status);
     /* assume all MS are the same size */  
     if (msgcode==CTRL_END || !msitr[0]->more()) {
cout<<"Slave "<<myrank<<" quitting"<<endl;
      break;
     } else if (msgcode==CTRL_SKIP) {
      /* skip to next timeslot */
      for(int cm=0; cm<mymscount; cm++) {
       (*msitr[cm])++;
      }
      tilex+=Data::TileSize;
      continue;
     }


     memset(Y,0,sizeof(double)*(size_t)iodata_vec[0].N*8*Mt*nsolbw);
     for (int nadmm=0; nadmm<Nadmm; nadmm++) {
      for (int nepch=0; nepch<nepochs; nepch++) {
        for (int nmb=0; nmb<minibatches; nmb++) {
/******************************* work on minibatch *****************************/
        if (!doBeam) {
         Data::loadDataMinibatch(msitr[0]->table(),iodata_vec[0],nmb,&iodata_vec[0].fratio);
        } else {
         Data::loadDataMinibatch(msitr[0]->table(),iodata_vec[0],beam_vec[0],nmb,&iodata_vec[0].fratio);
        }
        /* rescale u,v,w by 1/c NOT to wavelengths, that is done later in prediction */
        my_dscal(iodata_vec[0].Nbase*iodata_vec[0].tilesz,inv_c,iodata_vec[0].u);
        my_dscal(iodata_vec[0].Nbase*iodata_vec[0].tilesz,inv_c,iodata_vec[0].v);
        my_dscal(iodata_vec[0].Nbase*iodata_vec[0].tilesz,inv_c,iodata_vec[0].w);


        /* and set x[]=0 for flagged values */
        preset_flags_and_data(iodata_vec[0].Nbase*iodata_vec[0].tilesz,iodata_vec[0].flag,barr_vec[0],iodata_vec[0].x,Data::Nt);
#ifdef HAVE_CUDA
        /* update baseline flags */
        rearrange_baselines(iodata_vec[0].Nbase*iodata_vec[0].tilesz, barr_vec[0], hbb, Nt);
#endif

        /* precess source locations (also beam pointing) from J2000 to JAPP if we do any beam predictions,
           using first time slot as epoch */
        if (doBeam && !sources_precessed) {
         precess_source_locations(beam_vec[0].time_utc[iodata_vec[0].tilesz/2],carr_vec[0],M,&beam_vec[0].p_ra0,&beam_vec[0].p_dec0,Data::Nt);
         sources_precessed=1;
        }

    /****************** calibration **************************/
    /* coherency calculation need to be done per channel */
#ifndef HAVE_CUDA
    if (!doBeam) {
     precalculate_coherencies_multifreq(iodata_vec[0].u,iodata_vec[0].v,iodata_vec[0].w,coh_vec[0],iodata_vec[0].N,iodata_vec[0].Nbase*iodata_vec[0].tilesz,barr_vec[0],carr_vec[0],M,iodata_vec[0].freqs,iodata_vec[0].Nchan,iodata_vec[0].deltaf,iodata_vec[0].deltat,iodata_vec[0].dec0,Data::min_uvcut,Data::max_uvcut,Data::Nt);
    } else {
     //precalculate_coherencies_multifreq_withbeam(iodata.u,iodata.v,iodata.w,coh,iodata.N,iodata.Nbase*iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,iodata.deltaf,iodata.deltat,iodata.dec0,Data::min_uvcut,Data::max_uvcut,
  //beam.p_ra0,beam.p_dec0,iodata.freq0,beam.sx,beam.sy,beam.time_utc,iodata.tilesz,beam.Nelem,beam.xx,beam.yy,beam.zz,Data::Nt);
    }
#endif
#ifdef HAVE_CUDA
   if (GPUpredict) {
     /* note we need to use bandwith per channel here */
     precalculate_coherencies_multifreq_withbeam_gpu(iodata_vec[0].u,iodata_vec[0].v,iodata_vec[0].w,coh_vec[0],iodata_vec[0].N,iodata_vec[0].Nbase*iodata_vec[0].tilesz,barr_vec[0],carr_vec[0],M,iodata_vec[0].freqs,iodata_vec[0].Nchan,deltafch,iodata_vec[0].deltat,iodata_vec[0].dec0,Data::min_uvcut,Data::max_uvcut,
  beam_vec[0].p_ra0,beam_vec[0].p_dec0,iodata_vec[0].freq0,beam_vec[0].sx,beam_vec[0].sy,beam_vec[0].time_utc,iodata_vec[0].tilesz,beam_vec[0].Nelem,beam_vec[0].xx,beam_vec[0].yy,beam_vec[0].zz,doBeam,Data::Nt);
   } else {
    if (!doBeam) {
     precalculate_coherencies_multifreq(iodata_vec[0].u,iodata_vec[0].v,iodata_vec[0].w,coh_vec[0],iodata_vec[0].N,iodata_vec[0].Nbase*iodata_vec[0].tilesz,barr_vec[0],carr_vec[0],M,iodata_vec[0].freqs,iodata_vec[0].Nchan,iodata_vec[0].deltaf,iodata_vec[0].deltat,iodata_vec[0].dec0,Data::min_uvcut,Data::max_uvcut,Data::Nt);
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
        /* find B.Z for this freq, for all clusters */
        for (ci=0; ci<Mt; ci++) {
         memset(&z[8*iodata_vec[0].N*ci],0,sizeof(double)*(size_t)iodata_vec[0].N*8);
         for (int npp=0; npp<Npoly; npp++) {
          my_daxpy(8*iodata_vec[0].N, &Z[ci*8*iodata_vec[0].N*Npoly+npp*8*iodata_vec[0].N], B[ii*Npoly+npp], &z[8*iodata_vec[0].N*ci]);
         }
        }
        /* now z : 8NMt values = B Z */
        /* Y[ii*8*iodata.N*Mt] : 8NMt values */
        /* rhok[ii*Mt] : Mt values */
#ifdef HAVE_CUDA
        bfgsfit_minibatch_consensus(iodata_vec[0].u,iodata_vec[0].v,iodata_vec[0].w,&iodata_vec[0].xo[iodata_vec[0].Nbase*iodata_vec[0].tilesz*8*chanstart[ii]],iodata_vec[0].N,iodata_vec[0].Nbase,iodata_vec[0].tilesz,hbb,ptoclus,&coh_vec[0][M*iodata_vec[0].Nbase*iodata_vec[0].tilesz*4*chanstart[ii]],M,Mt,&iodata_vec[0].freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],&pfreq_vec[0][iodata_vec[0].N*8*Mt*ii],&Y[iodata_vec[0].N*8*Mt*ii],z,&rhok[ii*Mt],Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00[0],&res_01[0],&ptdata_array[ii],nmb,minibatches);
#else
        bfgsfit_minibatch_consensus(iodata_vec[0].u,iodata_vec[0].v,iodata_vec[0].w,&iodata_vec[0].xo[iodata_vec[0].Nbase*iodata_vec[0].tilesz*8*chanstart[ii]],iodata_vec[0].N,iodata_vec[0].Nbase,iodata_vec[0].tilesz,barr_vec[0],carr_vec[0],&coh_vec[0][M*iodata_vec[0].Nbase*iodata_vec[0].tilesz*4*chanstart[ii]],M,Mt,&iodata_vec[0].freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],&pfreq_vec[0][iodata_vec[0].N*8*Mt*ii],&Y[iodata_vec[0].N*8*Mt*ii],z,&rhok[ii*Mt],Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00[0],&res_01[0],&ptdata_array[ii],nmb,minibatches);
#endif
       res_0+=res_00[0];
       res_1+=res_01[0];
       resband[ii]=res_01[0];
       printf("%d: admm=%d epoch=%d minibatch=%d band=%d %lf %lf\n",myrank,nadmm,nepch,nmb,ii,res_00[0],res_01[0]);
      }
      /* find average residual over bands*/
      res_0/=(double)nsolbw;
      res_1/=(double)nsolbw;
      /* Examine bands where residual is far higher
         and exclude these bands when updating Z */
      for (ii=0; ii<nsolbw; ii++) {
        /* flag minibands with higher residual */
        fband[ii]=(resband[ii]>res_ratio*res_1?1:0);
      }
    /****************** end calibration **************************/

    
    /****************** ADMM update **************************/

      /* Y <- Y+ rho J */
      for (ii=0; ii<nsolbw; ii++) {
        if (!fband[ii]) { /* only update for good solutions */
         for (ci=0; ci<Mt; ci++) {
          my_daxpy(8*iodata_vec[0].N, &pfreq_vec[0][ii*8*iodata_vec[0].N*Mt+ci*8*iodata_vec[0].N], rhok[ii*Mt+ci], &Y[ii*8*iodata_vec[0].N*Mt+ci*8*iodata_vec[0].N]);
         }
        }
      }

      /* update Z : sum up B(Y+rho J) first*/
      for (ci=0; ci<Npoly; ci++) {
        my_dcopy(8*iodata_vec[0].N*Mt,Y,1,&z[ci*8*iodata_vec[0].N*Mt],1);
        my_dscal(8*iodata_vec[0].N*Mt,B[ci],&z[ci*8*iodata_vec[0].N*Mt]);
      }
      for (ii=1; ii<nsolbw; ii++) {
        if (!fband[ii]) { /* only update for good solutions */
        for(ci=0; ci<Npoly; ci++) {
          my_daxpy(8*iodata_vec[0].N*Mt, &Y[ii*8*iodata_vec[0].N*Mt], B[ii*Npoly+ci], &z[ci*8*iodata_vec[0].N*Mt]);
        }
        }
      }

      /* now add fed. avg. of Zavgxalpha, column by column (8NM values) to z
        at right location (shuffle). essentially the whole 8NMxNpoly values can be added in one step */
      /* ordering Z,Zavg: 8N Npoly x M,  z: 8N M Npoly x 1
        so transepose Npoly x M to M x Npoly from Z to z  */
      for (ci=0; ci<Mt; ci++) {
        for (cj=0; cj<Npoly; cj++) {
         my_daxpy(8*iodata_vec[0].N,&Zavg[8*iodata_vec[0].N*Npoly*ci+8*iodata_vec[0].N*cj],alpha,&z[8*iodata_vec[0].N*M*cj+8*iodata_vec[0].N*ci]);
        }
      }

      my_dcopy(iodata_vec[0].N*8*Npoly*Mt,Z,1,Zold,1);
      update_global_z_multi(Z,iodata_vec[0].N,Mt,Npoly,z,Bii,Data::Nt);

      my_daxpy(iodata_vec[0].N*8*Npoly*Mt,Z,-1.0,Zold);
      cout<<myrank<<": ADMM : "<<nadmm<<" dual residual="<<my_dnrm2(iodata_vec[0].N*8*Npoly*Mt,Zold)/sqrt((double)8*iodata_vec[0].N*Npoly*Mt)<<endl;

      /* update Y <- Y+rho*(J-B.Z), but already Y=Y+rho J
        so, only need to add -rho B.Z */
      for (ii=0; ii<nsolbw; ii++) {
        if (!fband[ii]) { /* only update for good solutions */
      for (ci=0; ci<Mt; ci++) {
       memset(&z[8*iodata_vec[0].N*ci],0,sizeof(double)*(size_t)iodata_vec[0].N*8);
       for (int npp=0; npp<Npoly; npp++) {
        my_daxpy(8*iodata_vec[0].N, &Z[ci*8*iodata_vec[0].N*Npoly+npp*8*iodata_vec[0].N], B[ii*Npoly+npp], &z[8*iodata_vec[0].N*ci]);
       }

       my_daxpy(8*iodata_vec[0].N, &z[ci*8*iodata_vec[0].N], -rhok[ii*Mt+ci], &Y[ii*8*iodata_vec[0].N*Mt+ci*8*iodata_vec[0].N]);
      }
      }
      }

    /****************** end ADMM update **************************/

        } /* minibatch */
      } /* epoch */
      
      /* send Z to master and get back the average */
      MPI_Send(Z, iodata_vec[0].N*8*Npoly*Mt, MPI_DOUBLE, 0,TAG_MSAUX, MPI_COMM_WORLD);
      MPI_Recv(Zavg, iodata_vec[0].N*8*Npoly*Mt, MPI_DOUBLE, 0,TAG_MSAUX, MPI_COMM_WORLD,&status);

      /* find error */
      my_dcopy(iodata_vec[0].N*8*Npoly*Mt,Z,1,Zold,1);
      my_daxpy(iodata_vec[0].N*8*Npoly*Mt,Zavg,-1.0,Zold);
      cout<<myrank<<":FEDA: "<<nadmm<<" dual residual="<<my_dnrm2(iodata_vec[0].N*8*Npoly*Mt,Zold)/sqrt((double)8*iodata_vec[0].N*Npoly*Mt)<<endl;
      //my_dcopy(iodata_vec[0].N*8*Npoly*Mt,Zavg,1,Z,1);
     } /* admm */

     if (start_iter) { start_iter=0; }

    /**********************************************************/
    /* also write back in minibatches */
    for (int nmb=0; nmb<minibatches; nmb++) {
     /* need to load the same data before calculating the residual */
     if (!doBeam) {
        Data::loadDataMinibatch(msitr[0]->table(),iodata_vec[0],nmb,&iodata_vec[0].fratio);
     } else {
        Data::loadDataMinibatch(msitr[0]->table(),iodata_vec[0],beam_vec[0],nmb,&iodata_vec[0].fratio);
     }
     /* calculate residual */
        /* rescale u,v,w by 1/c NOT to wavelengths, that is done later in prediction */
        my_dscal(iodata_vec[0].Nbase*iodata_vec[0].tilesz,inv_c,iodata_vec[0].u);
        my_dscal(iodata_vec[0].Nbase*iodata_vec[0].tilesz,inv_c,iodata_vec[0].v);
        my_dscal(iodata_vec[0].Nbase*iodata_vec[0].tilesz,inv_c,iodata_vec[0].w);

#ifndef HAVE_CUDA
      for (ii=0; ii<nsolbw; ii++) {
     if (!doBeam) {
       calculate_residuals_multifreq(iodata_vec[0].u,iodata_vec[0].v,iodata_vec[0].w,&pfreq_vec[0][iodata_vec[0].N*8*Mt*ii],&iodata_vec[0].xo[iodata_vec[0].Nbase*iodata_vec[0].tilesz*8*chanstart[ii]],iodata_vec[0].N,iodata_vec[0].Nbase,iodata_vec[0].tilesz,barr_vec[0],carr_vec[0],M,&iodata_vec[0].freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],iodata_vec[0].deltat,iodata_vec[0].dec0,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
     } else {
      calculate_residuals_multifreq_withbeam(iodata_vec[0].u,iodata_vec[0].v,iodata_vec[0].w,&pfreq_vec[0][iodata_vec[0].N*8*Mt*ii],&iodata_vec[0].xo[iodata_vec[0].Nbase*iodata_vec[0].tilesz*8*chanstart[ii]],iodata_vec[0].N,iodata_vec[0].Nbase,iodata_vec[0].tilesz,barr_vec[0],carr_vec[0],M,&iodata_vec[0].freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],iodata_vec[0].deltat,iodata_vec[0].dec0,
beam_vec[0].p_ra0,beam_vec[0].p_dec0,iodata_vec[0].freq0,beam_vec[0].sx,beam_vec[0].sy,beam_vec[0].time_utc,beam_vec[0].Nelem,beam_vec[0].xx,beam_vec[0].yy,beam_vec[0].zz,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
     }
      }
#endif
#ifdef HAVE_CUDA
      for (ii=0; ii<nsolbw; ii++) {
    if (GPUpredict) {
      calculate_residuals_multifreq_withbeam_gpu(iodata_vec[0].u,iodata_vec[0].v,iodata_vec[0].w,&pfreq_vec[0][iodata_vec[0].N*8*Mt*ii],&iodata_vec[0].xo[iodata_vec[0].Nbase*iodata_vec[0].tilesz*8*chanstart[ii]],iodata_vec[0].N,iodata_vec[0].Nbase,iodata_vec[0].tilesz,barr_vec[0],carr_vec[0],M,&iodata_vec[0].freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],iodata_vec[0].deltat,iodata_vec[0].dec0,
beam_vec[0].p_ra0,beam_vec[0].p_dec0,iodata_vec[0].freq0,beam_vec[0].sx,beam_vec[0].sy,beam_vec[0].time_utc,beam_vec[0].Nelem,beam_vec[0].xx,beam_vec[0].yy,beam_vec[0].zz,doBeam,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
    } else {
     if (!doBeam) {
       calculate_residuals_multifreq(iodata_vec[0].u,iodata_vec[0].v,iodata_vec[0].w,&pfreq_vec[0][iodata_vec[0].N*8*Mt*ii],&iodata_vec[0].xo[iodata_vec[0].Nbase*iodata_vec[0].tilesz*8*chanstart[ii]],iodata_vec[0].N,iodata_vec[0].Nbase,iodata_vec[0].tilesz,barr_vec[0],carr_vec[0],M,&iodata_vec[0].freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],iodata_vec[0].deltat,iodata_vec[0].dec0,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
     } else {
      calculate_residuals_multifreq_withbeam(iodata_vec[0].u,iodata_vec[0].v,iodata_vec[0].w,&pfreq_vec[0][iodata_vec[0].N*8*Mt*ii],&iodata_vec[0].xo[iodata_vec[0].Nbase*iodata_vec[0].tilesz*8*chanstart[ii]],iodata_vec[0].N,iodata_vec[0].Nbase,iodata_vec[0].tilesz,barr_vec[0],carr_vec[0],M,&iodata_vec[0].freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],iodata_vec[0].deltat,iodata_vec[0].dec0,
beam_vec[0].p_ra0,beam_vec[0].p_dec0,iodata_vec[0].freq0,beam_vec[0].sx,beam_vec[0].sy,beam_vec[0].time_utc,beam_vec[0].Nelem,beam_vec[0].xx,beam_vec[0].yy,beam_vec[0].zz,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
     }
    }
    }
#endif


    Data::writeDataMinibatch(msitr[0]->table(),iodata_vec[0],nmb);


    }
    for(int cm=0; cm<iodata_vec[0].Nms; cm++) {
      (*msitr[cm])++;
    }

     tilex+=Data::TileSize;

     for(int cm=0; cm<mymscount; cm++) {
     /* print solutions to file */
      for (cj=0; cj<iodata_vec[cm].N*8; cj++) {
       fprintf(sfp_vec[cm],"%d ",cj);
     for (ii=0; ii<nsolbw; ii++) {

       for (ci=M-1; ci>=0; ci--) {
         for (ck=0; ck<carr_vec[cm][ci].nchunk; ck++) {
          /* print solution */
          fprintf(sfp_vec[cm]," %e",pfreq_vec[cm][iodata_vec[0].N*8*Mt*ii+carr_vec[cm][ci].p[ck]+cj]);
         }
       }
     }
       fprintf(sfp_vec[cm],"\n");
      }
     }

   /* Reset solutions in mini-bands with bad residuals */
   for (ii=0; ii<nsolbw; ii++) {
       if (fband[ii]) {
        cout<<"Resetting solution for band "<<ii<<endl;
        memcpy(&pfreq_vec[0][iodata_vec[0].N*8*Mt*ii],pinit,(size_t)iodata_vec[0].N*8*Mt*sizeof(double));
        lbfgs_persist_reset(&ptdata_array[ii]);
       }
   }



     /* do some quality control */
    /* if residual has increased too much, or all are flagged (0 residual)
      or NaN
      reset solutions to original
      initial values : use residual at 1st ADMM */
    /* do not reset if initial residual is 0, because by def final one will be higher */
     for(int cm=0; cm<mymscount; cm++) {
      if (res_00[cm]!=0.0 && (res_01[cm]==0.0 || !isfinite(res_01[cm]) || res_01[cm]>res_ratio*res_prev[cm])) {
        cout<<"Resetting Solution "<<cm<<endl;
        /* reset solutions so next iteration has default initial values */
        for (ii=0; ii<nsolbw; ii++) {
         memcpy(&pfreq_vec[cm][iodata_vec[0].N*8*Mt*ii],pinit,(size_t)iodata_vec[cm].N*8*Mt*sizeof(double));
        }
        /* also assume iterations have restarted from scratch */
        start_iter=1;
        /* also forget min residual (otherwise will try to reset it always) */
        if (res_01[cm]!=0.0 && isfinite(res_01[cm])) {
         res_prev[cm]=res_01[cm];
        }
      } else if (res_01[cm]<res_prev[cm]) { /* only store the min value */
       res_prev[cm]=res_01[cm];
      }
    }
    end_time = time(0);
    elapsed_time = ((double) (end_time-start_time)) / 60.0;
     if (Data::verbose) {
      cout<<"nu="<<mean_nu<<endl;
     }
    if (Data::verbose) {
     for(int cm=0; cm<mymscount; cm++) {
      cout<<myrank<< ": Timeslot: "<<tilex<<" residual MS "<<cm<<": initial="<<res_00[cm]<<",final="<<res_01[cm]<<", Time spent="<<elapsed_time<<" minutes"<<endl;
     }
    }

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

     /* now send to master signal that we are ready for next data chunk */
     if (start_iter==1) {
      msgcode=CTRL_RESET;
     } else {
      msgcode=CTRL_DONE;
     }

     MPI_Send(&msgcode, 1, MPI_INT, 0,TAG_CTRL, MPI_COMM_WORLD);
    }

    /* free persistent memory */
    for (ii=0; ii<nsolbw; ii++) {
       lbfgs_persist_clear(&ptdata_array[ii]);
    }
    free(ptdata_array);


    for(int cm=0; cm<mymscount; cm++) {
     delete msitr[cm];
     delete msvector[cm];
    }

    for(int cm=0; cm<mymscount; cm++) {
     /* free data memory */
     if (!doBeam) {
      Data::freeData(iodata_vec[cm]);
     } else {
      Data::freeData(iodata_vec[cm],beam_vec[cm]);
     }
    }
 
    free(nchan);
    free(chanstart);

    /**********************************************************/

#ifdef HAVE_CUDA
   detach_gpu_from_thread(cbhandle,solver_handle);
   destroy_task_hist(&thst);
   free(hbb);
   free(ptoclus);
#endif


  exinfo_gaussian *exg;
  exinfo_disk *exd;
  exinfo_ring *exr;
  exinfo_shapelet *exs;

  for(int cm=0; cm<mymscount; cm++) {
  for (ci=0; ci<M; ci++) {
    free(carr_vec[cm][ci].ll);
    free(carr_vec[cm][ci].mm);
    free(carr_vec[cm][ci].nn);
    free(carr_vec[cm][ci].sI);
    free(carr_vec[cm][ci].sQ);
    free(carr_vec[cm][ci].sU);
    free(carr_vec[cm][ci].sV);
    free(carr_vec[cm][ci].p);
    free(carr_vec[cm][ci].ra);
    free(carr_vec[cm][ci].dec);
    for (cj=0; cj<carr_vec[cm][ci].N; cj++) {
     /* do a proper typecast before freeing */
     switch (carr_vec[cm][ci].stype[cj]) {
      case STYPE_GAUSSIAN:
        exg=(exinfo_gaussian*)carr_vec[cm][ci].ex[cj];
        if (exg) free(exg);
        break;
      case STYPE_DISK:
        exd=(exinfo_disk*)carr_vec[cm][ci].ex[cj];
        if (exd) free(exd);
        break;
      case STYPE_RING:
        exr=(exinfo_ring*)carr_vec[cm][ci].ex[cj];
        if (exr) free(exr);
        break;
      case STYPE_SHAPELET:
        exs=(exinfo_shapelet*)carr_vec[cm][ci].ex[cj];
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
    free(carr_vec[cm][ci].ex);
    free(carr_vec[cm][ci].stype);
    free(carr_vec[cm][ci].sI0);
    free(carr_vec[cm][ci].sQ0);
    free(carr_vec[cm][ci].sU0);
    free(carr_vec[cm][ci].sV0);
    free(carr_vec[cm][ci].f0);
    free(carr_vec[cm][ci].spec_idx);
    free(carr_vec[cm][ci].spec_idx1);
    free(carr_vec[cm][ci].spec_idx2);
  }
  free(carr_vec[cm]);
  free(barr_vec[cm]);
  free(p_vec[cm]);
  free(pfreq_vec[cm]);
  free(pm_vec[cm]);
  free(coh_vec[cm]);
  fclose(sfp_vec[cm]);
  }

  free(Z);
  free(Zold);
  free(Zavg);
  free(z);
  free(Y);
  free(B);
  free(Bii);
  free(rhok);
  free(resband);
  free(fband);

  free(pinit);
  /**********************************************************/

   cout<<"Done."<<endl;    
   return 0;
}
