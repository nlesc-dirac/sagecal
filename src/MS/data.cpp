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
#include "Radio.h"
#include "Dirac.h"
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/measures/Measures/UVWMachine.h>
#include <casacore/casa/Quanta.h>
#include <casacore/casa/Quanta/Quantum.h>

/* speed of light */
#ifndef CONST_C
#define CONST_C 299792458.0
#endif

using namespace casacore;

int Data::numChannels=1;
unsigned long int Data::numRows;


char *Data::TableName = NULL;
float Data::min_uvcut=0.0f;
float Data::max_uvcut=100e6;
float Data::max_uvtaper=0.0f;
String Data::DataField = "DATA";
String Data::OutField = "CORRECTED_DATA";
int Data::TileSize = 120;
int Data::Nt= 6; 
char *Data::SkyModel=NULL;
char *Data::Clusters=NULL;
int Data::format=1; /* defaut is LSM with 3rd order spectra */
double Data::nulow=2.0;
double Data::nuhigh=30.0;

int Data::max_emiter=3;
int Data::max_iter=2;
int Data::max_lbfgs=10;
int Data::lbfgs_m=7;
int Data::gpu_threads=128;
int Data::linsolv=1;
int Data::randomize=1;
int Data::whiten=0;
int Data::DoSim=0;
int Data::DoDiag=0;
int Data::doChan=0; /* if 1, solve for each channel in multi channel data */
int Data::doBeam=0; /* if >0, enable LOFAR beam model */
int Data::phaseOnly=0; /* if >0, enable phase only correction */
int Data::solver_mode=SM_RTR_OSRLM_RLBFGS; /* use RTR+LBFGS by default */
int Data::ccid=-99999;
double Data::rho=1e-9;
char *Data::solfile=NULL;
char *Data::initsolfile=NULL;
char *Data::ignorefile=NULL;
char *Data::MSlist=NULL;
char *Data::MSpattern=NULL;

/* stochastic calibration parameters */
int Data::stochastic_calib_epochs=0; /* if >1, how many epochs for running calib */
int Data::stochastic_calib_minibatches=1; /* how many minibatches data is split, if =1, minibatch=fullbatch */
int Data::stochastic_calib_bands=1; /* how many mini-bands the channels are split to, for bandpass calibration */
/* federated averaging, global - local constraint regularization factor */
double Data::federated_reg_alpha=0.1;

/* distributed sagecal parameters */
int Data::Nadmm=1;
int Data::Npoly=2;
int Data::PolyType=2;
double Data::admm_rho=5.0;
char *Data::admm_rho_file=NULL;
int Data::aadmm=0;

/* no upper limit, solve for all timeslots */
int Data::Nmaxtime=0;
/* skip starting time slots if given */
int Data::Nskip=0;
int Data::verbose=0; /* no verbose output */
int Data::mdl=0; /* no AIC/MDL calculation by default */
int Data::GPUpredict=0; /* use CPU for model calculation, if GPU not specified */
#ifdef HAVE_CUDA
int Data::heapsize=GPU_HEAP_SIZE; /* heap size in GPU (MB) to be used in malloc() */
#endif

int Data::servermode=-1; /* by default, no client-server mode */
char *Data::servername=NULL;
char *Data::portnumber=NULL; 

using namespace Data;

void
Data::readMSlist(char *fname, vector<string> *msnames) {
  cout<<"Reading "<<Data::MSlist<<endl;
     /* multiple MS */
     ifstream infile(fname);
    /* check if the file exists and readable */
    if(!infile.good()) {
     cout <<"File "<<Data::MSlist<<" does not exist."<<endl;
     exit(1);
    }
     string buffer;
     if (infile.is_open()) {
      while(infile.good()) {
       std::getline(infile,buffer);
       if (buffer.length()>0) {
       cout<<buffer<<endl;
        msnames->push_back(buffer);
       }
      }
     }
}

void 
Data::readAuxData(const char *fname, Data::IOData *data) {

    Table _t=Table(fname);
    Table _ant = Table(_t.keywordSet().asTable("ANTENNA"));
    ROScalarColumn<String> a1(_ant, "NAME");
    data->N=a1.nrow();
    data->Nbase=data->N*(data->N-1)/2;
    cout <<"Stations: "<<data->N<<" Baselines: "<<data->Nbase<<endl;

    ROScalarColumn<double> timeCol(_t, "INTERVAL"); 
    data->deltat=timeCol.get(0);
    data->totalt=(timeCol.nrow()+data->Nbase+data->N-1)/(data->Nbase+data->N);
    cout<<"Integration Time: "<<data->deltat<<" s,"<<" Total timeslots: "<<data->totalt<<endl;

    Table _field = Table(_t.keywordSet().asTable("FIELD"));
    ROArrayColumn<double> ref_dir(_field, "REFERENCE_DIR");
    Array<double> dir = ref_dir(0);
    double *c = dir.data();
    data->ra0=c[0];
    data->dec0=c[1];
    cout<<"Phase center ("<< c[0] << ", " << c[1] <<")"<<endl;

    //obtain the chanel freq information
    Table _freq = Table(_t.keywordSet().asTable("SPECTRAL_WINDOW"));
    ROArrayColumn<double> chan_freq(_freq, "CHAN_FREQ"); 
    data->Nchan=chan_freq.shape(0)[0];
    data->Nms=1;
   /* allocate memory */
   try { 
   data->u=new double[data->Nbase*data->tilesz];
   data->v=new double[data->Nbase*data->tilesz];
   data->w=new double[data->Nbase*data->tilesz];
   data->x=new double[8*data->Nbase*data->tilesz];
   data->xo=new double[8*data->Nbase*data->tilesz*data->Nchan];
   data->freqs=new double[data->Nchan];
   data->flag=new double[data->Nbase*data->tilesz];
   data->NchanMS=new int[data->Nms];
   } catch (const std::bad_alloc& e) {
     cout<<"Allocating memory for data failed. Quitting."<< e.what() << endl;
     exit(1);
   }
   data->NchanMS[0]=data->Nchan;

   /* copy freq */
   data->freq0=0.0;
   for (int ci=0; ci<data->Nchan; ci++) {
     data->freqs[ci]=chan_freq(0).data()[ci];
     data->freq0+=data->freqs[ci];
   }
   data->freq0/=(double)data->Nchan;
   /* need channel widths to calculate bandwidth */
   ROArrayColumn<double> chan_width(_freq, "CHAN_WIDTH"); 
   data->deltaf=(double)data->Nchan*(chan_width(0).data()[0]);
}

void 
Data::readAuxData(const char *fname, Data::IOData *data, Data::LBeam *binfo) {

    Table _t=Table(fname);
    Table _ant = Table(_t.keywordSet().asTable("ANTENNA"));
    ROScalarColumn<String> a1(_ant, "NAME");
    data->N=a1.nrow();
    data->Nbase=data->N*(data->N-1)/2;
    cout <<"Stations: "<<data->N<<" Baselines: "<<data->Nbase<<endl;

    ROScalarColumn<double> timeCol(_t, "INTERVAL"); 
    data->deltat=timeCol.get(0);
    data->totalt=(timeCol.nrow()+data->Nbase+data->N-1)/(data->Nbase+data->N);
    cout<<"Integration Time: "<<data->deltat<<" s,"<<" Total timeslots: "<<data->totalt<<endl;

    Table _field = Table(_t.keywordSet().asTable("FIELD"));
    ROArrayColumn<double> ref_dir(_field, "PHASE_DIR"); /* old REFERENCE_DIR */
    Array<double> dir = ref_dir(0);
    double *c = dir.data();
    data->ra0=c[0];
    data->dec0=c[1];
    cout<<"Phase center ("<< c[0] << ", " << c[1] <<")"<<endl;

    //obtain the chanel freq information
    Table _freq = Table(_t.keywordSet().asTable("SPECTRAL_WINDOW"));
    ROArrayColumn<double> chan_freq(_freq, "CHAN_FREQ"); 
    data->Nchan=chan_freq.shape(0)[0];
    data->Nms=1;
   /* allocate memory */
   try {
   data->u=new double[data->Nbase*data->tilesz];
   data->v=new double[data->Nbase*data->tilesz];
   data->w=new double[data->Nbase*data->tilesz];
   data->x=new double[8*data->Nbase*data->tilesz];
   data->xo=new double[8*data->Nbase*data->tilesz*data->Nchan];
   data->freqs=new double[data->Nchan];
   data->flag=new double[data->Nbase*data->tilesz];
   data->NchanMS=new int[data->Nms];
   } catch (const std::bad_alloc& e) {
     cout<<"Allocating memory for data failed. Quitting."<< e.what() << endl;
     exit(1);
   }
   data->NchanMS[0]=data->Nchan;

   /* copy freq */
   data->freq0=0.0;
   for (int ci=0; ci<data->Nchan; ci++) {
     data->freqs[ci]=chan_freq(0).data()[ci];
     data->freq0+=data->freqs[ci];
   }
   data->freq0/=(double)data->Nchan;
   /* need channel widths to calculate bandwidth */
   ROArrayColumn<double> chan_width(_freq, "CHAN_WIDTH"); 
   data->deltaf=(double)data->Nchan*(chan_width(0).data()[0]);

   /* UTC time */
   binfo->time_utc=new double[data->tilesz]; 
   /* no of elements in each station */
   binfo->Nelem=new int[data->N];
   /* positions of stations */
   binfo->sx=new double[data->N];
   binfo->sy=new double[data->N];
   binfo->sz=new double[data->N];
   /* coordinates of elements */
   binfo->xx=new double*[data->N];
   binfo->yy=new double*[data->N];
   binfo->zz=new double*[data->N];

   Table antfield;
   if(_t.keywordSet().fieldNumber("LOFAR_ANTENNA_FIELD") != -1) {
    antfield = Table(_t.keywordSet().asTable("LOFAR_ANTENNA_FIELD"));
   } else {
    char buff[2048]={0};
    sprintf(buff, "%s/LOFAR_ANTENNA_FIELD", fname);
    antfield=Table(buff);
   }
   ROArrayColumn<double> position(antfield, "POSITION"); 
   ROArrayColumn<double> offset(antfield, "ELEMENT_OFFSET"); 
   ROArrayColumn<double> coord(antfield, "COORDINATE_AXES"); 
   ROArrayColumn<bool> eflag(antfield, "ELEMENT_FLAG");
   ROArrayColumn<double> tileoffset(antfield, "TILE_ELEMENT_OFFSET");
   /* check if TILE_ELEMENT_OFFSET has any rows, of no rows present, 
      we know this is LBA */
   bool isHBA=tileoffset.hasContent(0);

   /* read positions, also setup memory for element coords */
   for (int ci=0; ci<data->N; ci++) {
     Array<double> _pos=position(ci);
     double *tx=_pos.data();
     binfo->sz[ci]=tx[2];

     MPosition stnpos(MVPosition(tx[0],tx[1],tx[2]),MPosition::ITRF);
     Array<double> _radpos=stnpos.getAngle("rad").getValue();
     tx=_radpos.data();

     binfo->sx[ci]=tx[0];
     binfo->sy[ci]=tx[1];
//cout<<"position "<<binfo->sx[ci]<<" "<<binfo->sy[ci]<<" "<<binfo->sz[ci]<<endl;
     binfo->Nelem[ci]=offset.shape(ci)[1];
   }


   if (isHBA) {
    double cones[16];
    for (int ci=0; ci<16; ci++) {
      cones[ci]=1.0;
    }
    double tempT[3*16];
    /* now read in element offsets, also transform them to local coordinates */
    for (int ci=0; ci<data->N; ci++) {
      Array<double> _off=offset(ci);
      double *off=_off.data();
      Array<double> _coord=coord(ci);
      double *coordmat=_coord.data();
      Array<bool> _eflag=eflag(ci);
      bool *ef=_eflag.data();
      Array<double> _toff=tileoffset(ci);
      double *toff=_toff.data();

/*
cout<<"A=["<<endl;
     for (int cj=0; cj<binfo->Nelem[ci]; cj++) {
cout<<off[3*cj]<<","<<off[3*cj+1]<<","<<off[3*cj+2]<<endl;
     }
cout<<"];"<<endl;
cout<<"T0=["<<endl;
     for (int cj=0; cj<16; cj++) {
cout<<toff[3*cj]<<","<<toff[3*cj+1]<<","<<toff[3*cj+2]<<endl;
     }
cout<<"];"<<endl;
cout<<"BT=["<<endl;
     for (int cj=0; cj<3; cj++) {
cout<<coordmat[3*cj]<<","<<coordmat[3*cj+1]<<","<<coordmat[3*cj+2]<<endl;
     }
cout<<"];"<<endl;
*/

      double *tempC=new double[3*binfo->Nelem[ci]];
      my_dgemm('T', 'N', binfo->Nelem[ci], 3, 3, 1.0, off, 3, coordmat, 3, 0.0, tempC, binfo->Nelem[ci]); 
      my_dgemm('T', 'N', 16, 3, 3, 1.0, toff, 3, coordmat, 3, 0.0, tempT, 16); 
  
/*
cout<<"C=["<<endl;
     for (int cj=0; cj<binfo->Nelem[ci]; cj++) {
cout<<tempC[cj]<<","<<tempC[cj+binfo->Nelem[ci]]<<","<<tempC[cj+2*binfo->Nelem[ci]]<<endl;
     }
cout<<"];"<<endl;
cout<<"T1=["<<endl;
     for (int cj=0; cj<16; cj++) {
cout<<tempT[cj]<<","<<tempT[cj+16]<<","<<tempT[cj+2*16]<<endl;
     }
cout<<"];"<<endl;
*/

      /* now inspect the element flag table to see if any of the dipoles are flagged */
      int fcount=0;
      for (int cj=0; cj<binfo->Nelem[ci]; cj++) {
       if (ef[2*cj]==1 || ef[2*cj+1]==1) {
        fcount++;
       } 
      }

//cout<<"%%Flagged "<<fcount<<endl;

      binfo->xx[ci]=new double[16*(binfo->Nelem[ci]-fcount)];
      binfo->yy[ci]=new double[16*(binfo->Nelem[ci]-fcount)];
      binfo->zz[ci]=new double[16*(binfo->Nelem[ci]-fcount)];
      /* copy unflagged coords, 16 times for each dipole */
      fcount=0;
      for (int cj=0; cj<binfo->Nelem[ci]; cj++) {
       if (!(ef[2*cj]==1 || ef[2*cj+1]==1)) {
        //binfo->xx[ci][fcount]=tempC[cj];
        //binfo->yy[ci][fcount]=tempC[cj+binfo->Nelem[ci]];
        //binfo->zz[ci][fcount]=tempC[cj+2*binfo->Nelem[ci]];
        my_dcopy(16,&tempT[0],1,&(binfo->xx[ci][fcount]),1);
        my_daxpy(16,cones,tempC[cj],&(binfo->xx[ci][fcount]));
        my_dcopy(16,&tempT[16],1,&(binfo->yy[ci][fcount]),1);
        my_daxpy(16,cones,tempC[cj+binfo->Nelem[ci]],&(binfo->yy[ci][fcount]));
        my_dcopy(16,&tempT[24],1,&(binfo->zz[ci][fcount]),1);
        my_daxpy(16,cones,tempC[cj+2*binfo->Nelem[ci]],&(binfo->zz[ci][fcount]));
        fcount+=16;
       }
      }
      binfo->Nelem[ci]=fcount;
/*
cout<<"%%Copied "<<fcount<<endl;
cout<<"D=["<<endl;
     for (int cj=0; cj<binfo->Nelem[ci]; cj++) {
cout<<binfo->xx[ci][cj]<<","<<binfo->yy[ci][cj]<<","<<binfo->zz[ci][cj]<<endl;
     }
cout<<"];"<<endl;
*/
      delete [] tempC;
    }
   } else { /* LBA */
    /* now read in element offsets, also transform them to local coordinates */
    for (int ci=0; ci<data->N; ci++) {
      Array<double> _off=offset(ci);
      double *off=_off.data();
      Array<double> _coord=coord(ci);
      double *coordmat=_coord.data();
      Array<bool> _eflag=eflag(ci);
      bool *ef=_eflag.data();

/*
cout<<"A=["<<endl;
     for (int cj=0; cj<binfo->Nelem[ci]; cj++) {
cout<<off[3*cj]<<","<<off[3*cj+1]<<","<<off[3*cj+2]<<endl;
     }
cout<<"];"<<endl;
cout<<"BT=["<<endl;
     for (int cj=0; cj<3; cj++) {
cout<<coordmat[3*cj]<<","<<coordmat[3*cj+1]<<","<<coordmat[3*cj+2]<<endl;
     }
cout<<"];"<<endl;
*/

      double *tempC=new double[3*binfo->Nelem[ci]];
      my_dgemm('T', 'N', binfo->Nelem[ci], 3, 3, 1.0, off, 3, coordmat, 3, 0.0, tempC, binfo->Nelem[ci]); 
  
/*
cout<<"C=["<<endl;
     for (int cj=0; cj<binfo->Nelem[ci]; cj++) {
cout<<tempC[cj]<<","<<tempC[cj+binfo->Nelem[ci]]<<","<<tempC[cj+2*binfo->Nelem[ci]]<<endl;
     }
cout<<"];"<<endl;
*/

      /* now inspect the element flag table to see if any of the dipoles are flagged */
      int fcount=0;
      for (int cj=0; cj<binfo->Nelem[ci]; cj++) {
       if (ef[2*cj]==1 || ef[2*cj+1]==1) {
        fcount++;
       } 
      }

//cout<<"%%Flagged "<<fcount<<endl;

      binfo->xx[ci]=new double[(binfo->Nelem[ci]-fcount)];
      binfo->yy[ci]=new double[(binfo->Nelem[ci]-fcount)];
      binfo->zz[ci]=new double[(binfo->Nelem[ci]-fcount)];
      /* copy unflagged coords for each dipole */
      fcount=0;
      for (int cj=0; cj<binfo->Nelem[ci]; cj++) {
       if (!(ef[2*cj]==1 || ef[2*cj+1]==1)) {
        binfo->xx[ci][fcount]=tempC[cj];
        binfo->yy[ci][fcount]=tempC[cj+binfo->Nelem[ci]];
        binfo->zz[ci][fcount]=tempC[cj+2*binfo->Nelem[ci]];
        fcount++;
       }
      }
      binfo->Nelem[ci]=fcount;
/*
cout<<"%%Copied "<<fcount<<endl;
cout<<"D=["<<endl;
     for (int cj=0; cj<binfo->Nelem[ci]; cj++) {
cout<<binfo->xx[ci][cj]<<","<<binfo->yy[ci][cj]<<","<<binfo->zz[ci][cj]<<endl;
     }
cout<<"];"<<endl;
*/
      delete [] tempC;
    }
   }

   /* read beam pointing direction (use Tile beam info: LBA also has it) */
   ROArrayColumn<double> point_dir(_field, "LOFAR_TILE_BEAM_DIR");
   Array<double> pdir = point_dir(0);
   double *pc = pdir.data();
   binfo->p_ra0=pc[0];
   binfo->p_dec0=pc[1];

   /* convert positions from xyz to longitude,latitude,height */
/*   double *longitude=new double[data->N];
   double *latitude=new double[data->N];
   double *height=new double[data->N];
   xyz2llh(binfo->sx,binfo->sy,binfo->sz,longitude,latitude,height,data->N);
   delete [] binfo->sx;
   delete [] binfo->sy;
   delete [] binfo->sz;
   binfo->sx=longitude;
   binfo->sy=latitude;
   binfo->sz=height;
*/
}


void 
Data::readAuxDataList(vector<string> msnames, Data::IOData *data) {
    /* read first filename */
    const char *fname=msnames[0].c_str();
    Table _t=Table(fname);
    //char buff[2048] = {0};
    //sprintf(buff, "%s/ANTENNA", fname);
    //Table _ant=Table(buff);
    Table _ant = Table(_t.keywordSet().asTable("ANTENNA"));
    ROScalarColumn<String> a1(_ant, "NAME");
    data->N=a1.nrow();
    data->Nbase=data->N*(data->N-1)/2;
    cout <<"Stations: "<<data->N<<" Baselines: "<<data->Nbase<<endl;

    ROScalarColumn<double> timeCol(_t, "INTERVAL"); 
    data->deltat=timeCol.get(0);
    data->totalt=(timeCol.nrow()+data->Nbase+data->N-1)/(data->Nbase+data->N);
    cout<<"Integration Time: "<<data->deltat<<" s,"<<" Total timeslots: "<<data->totalt<<endl;

    //sprintf(buff, "%s/FIELD", fname);
    //Table _field = Table(buff);
    Table _field = Table(_t.keywordSet().asTable("FIELD"));
    ROArrayColumn<double> ref_dir(_field, "REFERENCE_DIR");
    Array<double> dir = ref_dir(0);
    double *c = dir.data();
    data->ra0=c[0];
    data->dec0=c[1];
    cout<<"Phase center ("<< c[0] << ", " << c[1] <<")"<<endl;

    data->Nchan=0;
    data->Nms=msnames.size();
    data->NchanMS=new int[data->Nms];
    for (int cm=0; cm<data->Nms; cm++) {
     //obtain the chanel freq information
     fname=msnames[cm].c_str();
     Table _t1=Table(fname);
     //sprintf(buff, "%s/SPECTRAL_WINDOW", fname);
     //Table _freq = Table(buff);
     Table _freq = Table(_t1.keywordSet().asTable("SPECTRAL_WINDOW"));
     ROArrayColumn<double> chan_freq(_freq, "CHAN_FREQ"); 
     data->Nchan+=chan_freq.shape(0)[0];
     data->NchanMS[cm]=chan_freq.shape(0)[0];
    }
   cout<<"Total channels="<<data->Nchan<<endl;
 
   /* allocate memory */
   data->u=new double[data->Nbase*data->tilesz];
   data->v=new double[data->Nbase*data->tilesz];
   data->w=new double[data->Nbase*data->tilesz];
   data->x=new double[8*data->Nbase*data->tilesz];
   data->xo=new double[8*data->Nbase*data->tilesz*data->Nchan];
   data->freqs=new double[data->Nchan];
   data->flag=new double[data->Nbase*data->tilesz];

   /* copy freq */
   data->freq0=0.0;
   data->deltaf=0.0;
   int ck=0;
   for (int cm=0; cm<data->Nms; cm++) {
     fname=msnames[cm].c_str();
     Table _t1=Table(fname);
     //sprintf(buff, "%s/SPECTRAL_WINDOW", fname);
     //Table _freq = Table(buff);
     Table _freq = Table(_t1.keywordSet().asTable("SPECTRAL_WINDOW"));
     ROArrayColumn<double> chan_freq(_freq, "CHAN_FREQ"); 
     /* need channel widths to calculate bandwidth */
     ROArrayColumn<double> chan_width(_freq, "CHAN_WIDTH"); 
     for (int ci=0; ci<chan_freq.shape(0)[0]; ci++) {
     data->freqs[ck]=chan_freq(0).data()[ci];
     data->freq0+=data->freqs[ck++];
     data->deltaf+=(chan_width(0).data()[ci]);
    }
   }
   data->freq0/=(double)data->Nchan;
   cout<<"freq0="<<data->freq0/1e6<<endl;
   cout<<"deltaf="<<data->deltaf/1e6<<endl;
   for (ck=0; ck<data->Nchan;ck++){
    cout<<ck<<" "<<data->freqs[ck]<<endl;
   }
}


/* each time this is called read in data from MS, and format them as
  u,v,w: u,v,w coordinates (wavelengths) size Nbase*tilesz x 1 
  u,v,w are ordered with baselines, timeslots
  x: data to write size Nbase*8*tilesz x 1
  ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
  fratio: flagged data as a ratio to all available data
*/
void 
Data::loadData(Table ti, Data::IOData iodata, double *fratio) {

    /* sort input table by ant1 and ant2 */
    Block<String> iv1(3);
    iv1[0] = "TIME";
    iv1[1] = "ANTENNA1";
    iv1[2] = "ANTENNA2";
    Table t=ti.sort(iv1,Sort::Ascending);

    ROScalarColumn<int> a1(t, "ANTENNA1"), a2(t, "ANTENNA2");
    /* only read only access for input */
    ROArrayColumn<Complex> dataCol(t, Data::DataField);
    ROArrayColumn<double> uvwCol(t, "UVW"); 
    ROArrayColumn<bool> flagCol(t, "FLAG");

    /* check we get correct rows */
    int nrow=t.nrow();
    if(nrow<iodata.tilesz*iodata.Nbase-iodata.tilesz*iodata.N) {
      cout<<"Warning: Missing rows, got "<<nrow<<" expect "<<iodata.tilesz*iodata.Nbase<<" +- "<<iodata.tilesz*iodata.N<<". (probably the last time interval, so not a big issue)."<<endl;
    }
    int row0=0;
    /* tapering */
    bool dotaper=false;
    double invtaper=1.0;
    if (max_uvtaper>0.0f) {
      dotaper=true;
      /* taper in m */
      invtaper=iodata.freq0/((double)max_uvtaper*CONST_C);
    }
    /* counters for finding flagged data ratio */
    int countgood=0; int countbad=0;
    for(int row = 0; row < nrow && row0<iodata.tilesz*iodata.Nbase; row++) {
        uInt i = a1(row); //antenna1 
        uInt j = a2(row); //antenna2
        /* only work with cross correlations */
        if (i!=j) {
        Array<Complex> data = dataCol(row);
        Matrix<double> uvw = uvwCol(row);
        Array<bool> flag = flagCol(row);

        Complex cxx(0, 0);
        Complex cxy(0, 0);
        Complex cyx(0, 0);
        Complex cyy(0, 0);
        /* calculate sqrt(u^2+v^2) to select uv cuts */
        double *c = uvw.data();
        double uvd=sqrt(c[0]*c[0]+c[1]*c[1]);
        bool flag_uvcut=0;
        if (uvd<min_uvcut || uvd>max_uvcut) {
          flag_uvcut=true;
        } 
        double uvtaper=1.0;
        if (dotaper) {
         uvtaper=uvd*invtaper;
         if (uvtaper>1.0) {
          uvtaper=1.0;
         }
        }
        int nflag=0;
        for(int k = 0; k < iodata.Nchan; k++) {
           Complex *ptr = data[k].data();
           bool *flgptr=flag[k].data();
           if (!flgptr[0] && !flgptr[1] && !flgptr[2] && !flgptr[3]){
             cxx+=ptr[0];
             cxy+=ptr[1];
             cyx+=ptr[2];
             cyy+=ptr[3];
             nflag++; /* remeber unflagged datapoints */ 
           } 
        
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8]=ptr[0].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+1]=ptr[0].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+2]=ptr[1].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+3]=ptr[1].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+4]=ptr[2].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+5]=ptr[2].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+6]=ptr[3].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+7]=ptr[3].imag();
        }
        if (nflag>iodata.Nchan/2) { /* at least half channels should have good data */
         double invnflag=1.0/(double)nflag;
         cxx*=invnflag;
         cxy*=invnflag;
         cyx*=invnflag;
         cyy*=invnflag;
         if (dotaper) {
          cxx*=uvtaper;
          cxy*=uvtaper;
          cyx*=uvtaper;
          cyy*=uvtaper;
         }
         iodata.flag[row0]=0;
         countgood++;
        } else {
         if (!nflag) {
         /* all channels flagged, flag this row */
          iodata.flag[row0]=1;
          countbad++;
         } else {
          iodata.flag[row0]=2;
         }
        }
        iodata.u[row0]=c[0];
        iodata.v[row0]=c[1];
        iodata.w[row0]=c[2];
        if (flag_uvcut) {
            iodata.flag[row0]=2;
        }
        iodata.x[row0*8]=cxx.real();
        iodata.x[row0*8+1]=cxx.imag();
        iodata.x[row0*8+2]=cxy.real();
        iodata.x[row0*8+3]=cxy.imag();
        iodata.x[row0*8+4]=cyx.real();
        iodata.x[row0*8+5]=cyx.imag();
        iodata.x[row0*8+6]=cyy.real();
        iodata.x[row0*8+7]=cyy.imag();

       row0++;
      }
    }
    /* now if there is a tail of empty data remaining, flag them */
    if (row0<iodata.tilesz*iodata.Nbase) {
      for(int row = row0; row<iodata.tilesz*iodata.Nbase; row++) {
        iodata.flag[row]=1;

      }
      /* set uvw and data to 0 to eliminate any funny business */
      memset(&iodata.u[row0],0,sizeof(double)*(size_t)(iodata.tilesz*iodata.Nbase-row0));
      memset(&iodata.v[row0],0,sizeof(double)*(size_t)(iodata.tilesz*iodata.Nbase-row0));
      memset(&iodata.w[row0],0,sizeof(double)*(size_t)(iodata.tilesz*iodata.Nbase-row0));
      memset(&iodata.x[8*row0],0,sizeof(double)*(size_t)8*(iodata.tilesz*iodata.Nbase-row0));

      for(int k = 0; k < iodata.Nchan; k++) {
       memset(&iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8],0,sizeof(double)*(size_t)8*(iodata.tilesz*iodata.Nbase-row0));
      }
    }
    /* flagged data / total usable data, not counting excluded baselines */
    if (countgood+countbad>0) {
     *fratio=(double)countbad/(double)(countgood+countbad);
    } else {
     *fratio=1.0;
    }
}

/* each time this is called read in data from MS, and format them as
  u,v,w: u,v,w coordinates (wavelengths) size Nbase*tilesz x 1 
  u,v,w are ordered with baselines, timeslots
  x: data to write size Nbase*8*tileze x 1
  ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots

  time_utc: UTC time, 1 per each Nbase
  fratio: flagged data as a ratio to all available data
*/
void 
Data::loadData(Table ti, Data::IOData iodata, LBeam binfo, double *fratio) {

    /* sort input table by ant1 and ant2 */
    Block<String> iv1(3);
    iv1[0] = "TIME";
    iv1[1] = "ANTENNA1";
    iv1[2] = "ANTENNA2";
    Table t=ti.sort(iv1,Sort::Ascending);

    ROScalarColumn<int> a1(t, "ANTENNA1"), a2(t, "ANTENNA2");
    /* only read only access for input */
    ROArrayColumn<Complex> dataCol(t, Data::DataField);
    ROArrayColumn<double> uvwCol(t, "UVW"); 
    ROArrayColumn<bool> flagCol(t, "FLAG");
    ROScalarColumn<double> tut(t,"TIME");

    /* check we get correct rows */
    int nrow=t.nrow();
    if(nrow<iodata.tilesz*iodata.Nbase-iodata.tilesz*iodata.N) {
      cout<<"Warning: Missing rows, got "<<nrow<<" expect "<<iodata.tilesz*iodata.Nbase<<" +- "<<iodata.tilesz*iodata.N<<". (probably the last time interval, so not a big issue)."<<endl;
    }
    int row0=0;
    int rowt=0;
    /* tapering */
    bool dotaper=false;
    double invtaper=1.0;
    if (max_uvtaper>0.0f) {
      dotaper=true;
      /* taper in m */
      invtaper=iodata.freq0/((double)max_uvtaper*CONST_C);
    }
    /* counters for finding flagged data ratio */
    int countgood=0; int countbad=0;
    for(int row = 0; row < nrow && row0<iodata.tilesz*iodata.Nbase; row++) {
        uInt i = a1(row); //antenna1 
        uInt j = a2(row); //antenna2
        if (!i && !j) {/* use baseline 0-0 to extract time */
         double tt=tut(row);
         /* convert MJD (s) to JD (days) */
         binfo.time_utc[rowt++]=(tt/86400.0+2400000.5); /* no +0.5 added */
        }
        /* only work with cross correlations */
        if (i!=j) {
        Array<Complex> data = dataCol(row);
        Matrix<double> uvw = uvwCol(row);
        Array<bool> flag = flagCol(row);

        Complex cxx(0, 0);
        Complex cxy(0, 0);
        Complex cyx(0, 0);
        Complex cyy(0, 0);
        /* calculate sqrt(u^2+v^2) to select uv cuts */
        double *c = uvw.data();
        double uvd=sqrt(c[0]*c[0]+c[1]*c[1]);
        bool flag_uvcut=0;
        if (uvd<min_uvcut || uvd>max_uvcut) {
          flag_uvcut=true;
        } 
        double uvtaper=1.0;
        if (dotaper) {
         uvtaper=uvd*invtaper;
         if (uvtaper>1.0) {
          uvtaper=1.0;
         }
        }
        int nflag=0;
        for(int k = 0; k < iodata.Nchan; k++) {
           Complex *ptr = data[k].data();
           bool *flgptr=flag[k].data();
           if (!flgptr[0] && !flgptr[1] && !flgptr[2] && !flgptr[3]){
             cxx+=ptr[0];
             cxy+=ptr[1];
             cyx+=ptr[2];
             cyy+=ptr[3];
             nflag++; /* remeber unflagged datapoints */ 
           } 
        
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8]=ptr[0].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+1]=ptr[0].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+2]=ptr[1].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+3]=ptr[1].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+4]=ptr[2].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+5]=ptr[2].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+6]=ptr[3].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+7]=ptr[3].imag();
        }
        if (nflag>iodata.Nchan/2) { /* at least half channels should have good data */
         double invnflag=1.0/(double)nflag;
         cxx*=invnflag;
         cxy*=invnflag;
         cyx*=invnflag;
         cyy*=invnflag;
         if (dotaper) {
          cxx*=uvtaper;
          cxy*=uvtaper;
          cyx*=uvtaper;
          cyy*=uvtaper;
         }
         iodata.flag[row0]=0;
         countgood++;
        } else {
         if (!nflag) {
         /* all channels flagged, flag this row */
          iodata.flag[row0]=1;
          countbad++;
         } else {
          iodata.flag[row0]=2;
         }
        }
        iodata.u[row0]=c[0];
        iodata.v[row0]=c[1];
        iodata.w[row0]=c[2];
        if (flag_uvcut) {
            iodata.flag[row0]=2;
        }
        iodata.x[row0*8]=cxx.real();
        iodata.x[row0*8+1]=cxx.imag();
        iodata.x[row0*8+2]=cxy.real();
        iodata.x[row0*8+3]=cxy.imag();
        iodata.x[row0*8+4]=cyx.real();
        iodata.x[row0*8+5]=cyx.imag();
        iodata.x[row0*8+6]=cyy.real();
        iodata.x[row0*8+7]=cyy.imag();

       row0++;
      }
    }
    /* now if there is a tail of empty data remaining, flag them */
    if (row0<iodata.tilesz*iodata.Nbase) {
      for(int row = row0; row<iodata.tilesz*iodata.Nbase; row++) {
        iodata.flag[row]=1;

      }
      /* set uvw and data to 0 to eliminate any funny business */
      memset(&iodata.u[row0],0,sizeof(double)*(size_t)(iodata.tilesz*iodata.Nbase-row0));
      memset(&iodata.v[row0],0,sizeof(double)*(size_t)(iodata.tilesz*iodata.Nbase-row0));
      memset(&iodata.w[row0],0,sizeof(double)*(size_t)(iodata.tilesz*iodata.Nbase-row0));
      memset(&iodata.x[8*row0],0,sizeof(double)*(size_t)8*(iodata.tilesz*iodata.Nbase-row0));

      for(int k = 0; k < iodata.Nchan; k++) {
       memset(&iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8],0,sizeof(double)*(size_t)8*(iodata.tilesz*iodata.Nbase-row0));
      }
    }
    /* flagged data / total usable data, not counting excluded baselines */
    if (countgood+countbad>0) {
     *fratio=(double)countbad/(double)(countgood+countbad);
    } else {
     *fratio=1.0;
    }
}


/* each time this is called read in data from MS, and format them as
  u,v,w: u,v,w coordinates (wavelengths) size Nbase*tilesz x 1 
  u,v,w are ordered with baselines, timeslots
  x: data to write size Nbase*8*tileze x 1
  ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
  fratio: flagged data as a ratio to all available data
*/
void 
Data::loadDataList(vector<MSIter*> msitr, Data::IOData iodata, double *fratio) {
    Table ti=msitr[0]->table();
    /* sort input table by ant1 and ant2 */
    Block<String> iv1(3);
    iv1[0] = "TIME";
    iv1[1] = "ANTENNA1";
    iv1[2] = "ANTENNA2";
    Table t=ti.sort(iv1,Sort::Ascending);

    ROScalarColumn<int> a1(t, "ANTENNA1"), a2(t, "ANTENNA2");
    /* only read only access for input */
    ROArrayColumn<double> uvwCol(t, "UVW"); 

    /* check we get correct rows */
    int nrow=t.nrow();
    if(nrow-iodata.N*iodata.tilesz>iodata.tilesz*iodata.Nbase) {
      cout<<"Error in rows"<<endl;
    }
  vector<ROArrayColumn<Complex>* > dataCols(iodata.Nms);
  vector<ROArrayColumn<bool>* > flagCols(iodata.Nms);
  for (int cm=0; cm<iodata.Nms;cm++) { 
    Table tti=(msitr[cm]->table());
    Table *tt=new Table(tti.sort(iv1,Sort::Ascending));
    dataCols[cm] = new  ROArrayColumn<Complex>(*tt,Data::DataField);
    flagCols[cm] = new  ROArrayColumn<bool>(*tt,"FLAG");
  }
    /* tapering */
    bool dotaper=false;
    double invtaper=1.0;
    if (max_uvtaper>0.0f) {
      dotaper=true;
      /* taper in m */
      invtaper=iodata.freq0/((double)max_uvtaper*CONST_C);
    }

    /* counters for finding flagged data ratio */
    int countgood=0; int countbad=0;

    int row0=0;
    for(int row = 0; row < nrow; row++) {
        uInt i = a1(row); //antenna1 
        uInt j = a2(row); //antenna2
        /* only work with cross correlations */
        if (i!=j) {
        Matrix<double> uvw = uvwCol(row);

        Complex cxx(0, 0);
        Complex cxy(0, 0);
        Complex cyx(0, 0);
        Complex cyy(0, 0);
        /* calculate sqrt(u^2+v^2) to select uv cuts */
        double *c = uvw.data();
        double uvd=sqrt(c[0]*c[0]+c[1]*c[1]);
        bool flag_uvcut=0;
        if (uvd<min_uvcut || uvd>max_uvcut) {
          flag_uvcut=true;
        } 
        int nflag=0;
        double uvtaper=1.0;
        if (dotaper) {
         uvtaper=uvd*invtaper;
         if (uvtaper>1.0) {
          uvtaper=1.0;
         }
        }

  int chanoff=0;
  for (int cm=0; cm<iodata.Nms;cm++) { 
        Array<Complex> data = (*(dataCols[cm]))(row);
        Array<bool> flag = (*(flagCols[cm]))(row);
        for(int k = 0; k < iodata.NchanMS[cm]; k++) {
           Complex *ptr = data[k].data();
           bool *flgptr=flag[k].data();
           if (!flgptr[0] && !flgptr[1] && !flgptr[2] && !flgptr[3]){
             cxx+=ptr[0];
             cxy+=ptr[1];
             cyx+=ptr[2];
             cyy+=ptr[3];
             nflag++; /* remeber unflagged datapoints */ 
           } 
        
           iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8]=ptr[0].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+1]=ptr[0].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+2]=ptr[1].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+3]=ptr[1].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+4]=ptr[2].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+5]=ptr[2].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+6]=ptr[3].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+7]=ptr[3].imag();
           chanoff++;
        }
   }

        if (nflag>iodata.Nchan/2) {/* at least half channels should have good data */
         double invnflag=1.0/(double)nflag;
         cxx*=invnflag;
         cxy*=invnflag;
         cyx*=invnflag;
         cyy*=invnflag;
         if (dotaper) {
          cxx*=uvtaper;
          cxy*=uvtaper;
          cyx*=uvtaper;
          cyy*=uvtaper;
         }
         iodata.flag[row0]=0;
         countgood++;
        } else {
         if (!nflag) {
         /* all channels flagged, flag this row */
          iodata.flag[row0]=1;
          countbad++;
         } else {
          iodata.flag[row0]=2;
         }
        }
        iodata.u[row0]=c[0];
        iodata.v[row0]=c[1];
        iodata.w[row0]=c[2];
        if (flag_uvcut) {
            iodata.flag[row0]=2;
        }
        iodata.x[row0*8]=cxx.real();
        iodata.x[row0*8+1]=cxx.imag();
        iodata.x[row0*8+2]=cxy.real();
        iodata.x[row0*8+3]=cxy.imag();
        iodata.x[row0*8+4]=cyx.real();
        iodata.x[row0*8+5]=cyx.imag();
        iodata.x[row0*8+6]=cyy.real();
        iodata.x[row0*8+7]=cyy.imag();

       row0++;
      }
    }
    /* now if there is a tail of empty data remaining, flag them */
    if (row0<iodata.tilesz*iodata.Nbase) {
      for(int row = row0; row<iodata.tilesz*iodata.Nbase; row++) {
        iodata.flag[row]=1;
      }
    }
    for (int cm=0; cm<iodata.Nms;cm++) {
     delete dataCols[cm];
     delete flagCols[cm];
    }
    /* flagged data / total usable data, not counting excluded baselines */
    if (countgood+countbad>0) {
     *fratio=(double)countbad/(double)(countgood+countbad);
    } else {
     *fratio=1.0;
    }
}


/* load data in mini-batches
 minibatch: 0...total_minibatches_per_epoch-1,
 tilesz: how many time slots are included in one minibatch? 1...tile_size
 Data::TileSize gives the full time slots
 skip until correct batch is reached
 assume iodata can store only tileszxNbaseline rows (x Nchan)
 Note: no average over channels of data is calculated */
void 
Data::loadDataMinibatch(Table ti, Data::IOData iodata, int minibatch, double *fratio) {

    /* first iterate to the right minibatch */ 
    Block<String> ivl(1); ivl[0]="TIME"; 
    TableIterator tit(ti,ivl);
    /* till which timeslot should we iterate ? */
    int tillts=minibatch*iodata.tilesz;
    int ttime=0;
    while(!tit.pastEnd() && ttime<tillts) {
      tit.next();
      ttime++;
    }

    /* sort input table by ant1 and ant2 */
    Block<String> iv1(2);
    iv1[0] = "ANTENNA1";
    iv1[1] = "ANTENNA2";

    /* how many timeslots to read now, if we have reached a valid row */
    int tmb=0;
    int rowoffset=0;
    /* counters for finding flagged data ratio */
    int countgood=0; int countbad=0;

    while(!tit.pastEnd() && tmb<iodata.tilesz) {

    Table t=tit.table().sort(iv1,Sort::Ascending);

    ROScalarColumn<int> a1(t, "ANTENNA1"), a2(t, "ANTENNA2");
    /* only read only access for input */
    ROArrayColumn<Complex> dataCol(t, Data::DataField);
    ROArrayColumn<double> uvwCol(t, "UVW"); 
    ROArrayColumn<bool> flagCol(t, "FLAG");

    /* check we get correct rows */
    int nrow=t.nrow();
    int row0=rowoffset; /* begin with right offset in iodata */
    for(int row = 0; row < nrow && row0<iodata.tilesz*iodata.Nbase; row++) {
        uInt i = a1(row); //antenna1 
        uInt j = a2(row); //antenna2
        /* only work with cross correlations */
        if (i!=j) {
        Array<Complex> data = dataCol(row);
        Matrix<double> uvw = uvwCol(row);
        Array<bool> flag = flagCol(row);

        /* calculate sqrt(u^2+v^2) to select uv cuts */
        double *c = uvw.data();
        double uvd=sqrt(c[0]*c[0]+c[1]*c[1]);
        bool flag_uvcut=0;
        if (uvd<min_uvcut || uvd>max_uvcut) {
          flag_uvcut=true;
        } 
        int nflag=0;
        for(int k = 0; k < iodata.Nchan; k++) {
           Complex *ptr = data[k].data();
           bool *flgptr=flag[k].data();
           if (!flgptr[0] && !flgptr[1] && !flgptr[2] && !flgptr[3]){
             nflag++; /* remeber unflagged datapoints */ 
           } 
        
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8]=ptr[0].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+1]=ptr[0].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+2]=ptr[1].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+3]=ptr[1].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+4]=ptr[2].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+5]=ptr[2].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+6]=ptr[3].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+7]=ptr[3].imag();
        }
        if (nflag>iodata.Nchan/2) { /* at least half channels should have good data */
         iodata.flag[row0]=0;
         countgood++;
        } else {
         if (!nflag) {
         /* all channels flagged, flag this row */
          iodata.flag[row0]=1;
          countbad++;
         } else {
          iodata.flag[row0]=2;
         }
        }
        iodata.u[row0]=c[0];
        iodata.v[row0]=c[1];
        iodata.w[row0]=c[2];
        if (flag_uvcut) {
            iodata.flag[row0]=2;
        }
       row0++;
      }
    }

     tmb++;
     rowoffset=row0;
     /* go to next timeslot */
     tit.next();
    }

    /* now if there is a tail of empty data remaining, flag them */
    if (rowoffset<iodata.tilesz*iodata.Nbase) {
      for(int row = rowoffset; row<iodata.tilesz*iodata.Nbase; row++) {
        iodata.flag[row]=1;
      }
      /* set uvw and data to 0 to eliminate any funny business */
      memset(&iodata.u[rowoffset],0,sizeof(double)*(size_t)(iodata.tilesz*iodata.Nbase-rowoffset));
      memset(&iodata.v[rowoffset],0,sizeof(double)*(size_t)(iodata.tilesz*iodata.Nbase-rowoffset));
      memset(&iodata.w[rowoffset],0,sizeof(double)*(size_t)(iodata.tilesz*iodata.Nbase-rowoffset));

      for(int k = 0; k < iodata.Nchan; k++) {
       memset(&iodata.xo[iodata.Nbase*iodata.tilesz*8*k+rowoffset*8],0,sizeof(double)*(size_t)8*(iodata.tilesz*iodata.Nbase-rowoffset));
      }
    }


    /* flagged data / total usable data, not counting excluded baselines */
    if (countgood+countbad>0) {
     *fratio=(double)countbad/(double)(countgood+countbad);
    } else {
     *fratio=1.0;
    }

}


void 
Data::loadDataMinibatch(Table ti, Data::IOData iodata, LBeam binfo, int minibatch, double *fratio) {

    /* first iterate to the right minibatch */
    Block<String> ivl(1); ivl[0]="TIME";
    TableIterator tit(ti,ivl);
    /* till which timeslot should we iterate ? */
    int tillts=minibatch*iodata.tilesz;
    int ttime=0;
    while(!tit.pastEnd() && ttime<tillts) {
      tit.next();
      ttime++;
    }

    /* sort input table by ant1 and ant2 */
    Block<String> iv1(2);
    iv1[0] = "ANTENNA1";
    iv1[1] = "ANTENNA2";

    /* how many timeslots to read now, if we have reached a valid row */
    int tmb=0;
    int rowoffset=0;
    int rowtoffset=0;
    /* counters for finding flagged data ratio */
    int countgood=0; int countbad=0;

    while(!tit.pastEnd() && tmb<iodata.tilesz) {

    Table t=tit.table().sort(iv1,Sort::Ascending);

    ROScalarColumn<int> a1(t, "ANTENNA1"), a2(t, "ANTENNA2");
    /* only read only access for input */
    ROArrayColumn<Complex> dataCol(t, Data::DataField);
    ROArrayColumn<double> uvwCol(t, "UVW"); 
    ROArrayColumn<bool> flagCol(t, "FLAG");
    ROScalarColumn<double> tut(t,"TIME");

    /* check we get correct rows */
    int nrow=t.nrow();
    int row0=rowoffset;
    int rowt=rowtoffset;
    for(int row = 0; row < nrow && row0<iodata.tilesz*iodata.Nbase; row++) {
        uInt i = a1(row); //antenna1 
        uInt j = a2(row); //antenna2
        if (!i && !j) {/* use baseline 0-0 to extract time */
         double tt=tut(row);
         /* convert MJD (s) to JD (days) */
         binfo.time_utc[rowt++]=(tt/86400.0+2400000.5); /* no +0.5 added */
        }
        /* only work with cross correlations */
        if (i!=j) {
        Array<Complex> data = dataCol(row);
        Matrix<double> uvw = uvwCol(row);
        Array<bool> flag = flagCol(row);

        /* calculate sqrt(u^2+v^2) to select uv cuts */
        double *c = uvw.data();
        double uvd=sqrt(c[0]*c[0]+c[1]*c[1]);
        bool flag_uvcut=0;
        if (uvd<min_uvcut || uvd>max_uvcut) {
          flag_uvcut=true;
        } 
        int nflag=0;
        for(int k = 0; k < iodata.Nchan; k++) {
           Complex *ptr = data[k].data();
           bool *flgptr=flag[k].data();
           if (!flgptr[0] && !flgptr[1] && !flgptr[2] && !flgptr[3]){
             nflag++; /* remeber unflagged datapoints */ 
           } 
        
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8]=ptr[0].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+1]=ptr[0].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+2]=ptr[1].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+3]=ptr[1].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+4]=ptr[2].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+5]=ptr[2].imag();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+6]=ptr[3].real();
           iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+7]=ptr[3].imag();
        }
        if (nflag>iodata.Nchan/2) { /* at least half channels should have good data */
         iodata.flag[row0]=0;
         countgood++;
        } else {
         if (!nflag) {
         /* all channels flagged, flag this row */
          iodata.flag[row0]=1;
          countbad++;
         } else {
          iodata.flag[row0]=2;
         }
        }
        iodata.u[row0]=c[0];
        iodata.v[row0]=c[1];
        iodata.w[row0]=c[2];
        if (flag_uvcut) {
            iodata.flag[row0]=2;
        }

       row0++;
      }
    }

     tmb++;
     rowoffset=row0;
     rowtoffset=rowt;
     /* go to next timeslot */
     tit.next();

    }

    /* now if there is a tail of empty data remaining, flag them */
    if (rowoffset<iodata.tilesz*iodata.Nbase) {
      for(int row = rowoffset; row<iodata.tilesz*iodata.Nbase; row++) {
        iodata.flag[row]=1;
      }
      /* set uvw and data to 0 to eliminate any funny business */
      memset(&iodata.u[rowoffset],0,sizeof(double)*(size_t)(iodata.tilesz*iodata.Nbase-rowoffset));
      memset(&iodata.v[rowoffset],0,sizeof(double)*(size_t)(iodata.tilesz*iodata.Nbase-rowoffset));
      memset(&iodata.w[rowoffset],0,sizeof(double)*(size_t)(iodata.tilesz*iodata.Nbase-rowoffset));

      for(int k = 0; k < iodata.Nchan; k++) {
       memset(&iodata.xo[iodata.Nbase*iodata.tilesz*8*k+rowoffset*8],0,sizeof(double)*(size_t)8*(iodata.tilesz*iodata.Nbase-rowoffset));
      }
    }

    /* flagged data / total usable data, not counting excluded baselines */
    if (countgood+countbad>0) {
     *fratio=(double)countbad/(double)(countgood+countbad);
    } else {
     *fratio=1.0;
    }

}




void 
Data::writeData(Table ti, Data::IOData iodata) {

    /* sort input table by ant1 and ant2 */
    Block<String> iv1(3);
    iv1[0] = "TIME";
    iv1[1] = "ANTENNA1";
    iv1[2] = "ANTENNA2";
    Table t=ti.sort(iv1,Sort::Ascending);

    ROScalarColumn<int> a1(t, "ANTENNA1"), a2(t, "ANTENNA2");
    /* writable access for output */
    ArrayColumn<Complex> dataCol(t, Data::OutField);

    /* check we get correct rows */
    int nrow=t.nrow();
    if(nrow-iodata.N*iodata.tilesz>iodata.tilesz*iodata.Nbase) {
      cout<<"Warning: Missing rows, got "<<nrow<<" expect "<<iodata.tilesz*iodata.Nbase<<" +- "<<iodata.tilesz*iodata.N<<". (probably the last time interval, so not a big issue)."<<endl;
    }
    //cout<<"Table rows "<<nrow<<" Data rows "<<iodata.tilesz*iodata.Nbase+iodata.tilesz*iodata.N<<endl;
    int row0=0;
    IPosition pos(2,4,iodata.Nchan);
    for(int row = 0; row < nrow; row++) {
        uInt i = a1(row); //antenna1 
        uInt j = a2(row); //antenna2
        /* only work with cross correlations */
        if (i!=j) {
        Array<Complex> data = dataCol(row);
        for(int k = 0; k < iodata.Nchan; k++) {
           pos(0)=0;pos(1)=k;
           data(pos)=Complex(iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8],iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+1]);
           pos(0)=1;pos(1)=k;
           data(pos)=Complex(iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+2],iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+3]);
           pos(0)=2;pos(1)=k;
           data(pos)=Complex(iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+4],iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+5]);
           pos(0)=3;pos(1)=k;
           data(pos)=Complex(iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+6],iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+7]);
       }

       row0++;
       dataCol.put(row,data); // copy to output
      }
    }
}

void 
Data::writeDataList(vector<MSIter*> msitr, Data::IOData iodata) {
    Table ti=msitr[0]->table();
    /* sort input table by ant1 and ant2 */
    Block<String> iv1(3);
    iv1[0] = "TIME";
    iv1[1] = "ANTENNA1";
    iv1[2] = "ANTENNA2";
    Table t=ti.sort(iv1,Sort::Ascending);

    ROScalarColumn<int> a1(t, "ANTENNA1"), a2(t, "ANTENNA2");

    /* check we get correct rows */
    int nrow=t.nrow();
    if(nrow-iodata.N*iodata.tilesz>iodata.tilesz*iodata.Nbase) {
      cout<<"Error in rows"<<endl;
    }
    vector<ArrayColumn<Complex>* > dataCols(iodata.Nms);
    for (int cm=0; cm<iodata.Nms;cm++) {
     Table tti=(msitr[cm]->table());
     Table *tt=new Table(tti.sort(iv1,Sort::Ascending));
     dataCols[cm] = new  ArrayColumn<Complex>(*tt,Data::OutField);
   }

    int row0=0;
    for(int row = 0; row < nrow; row++) {
        uInt i = a1(row); //antenna1 
        uInt j = a2(row); //antenna2
        /* only work with cross correlations */
        if (i!=j) {
  int chanoff=0;
  for (int cm=0; cm<iodata.Nms;cm++) {
        Array<Complex> data = (*(dataCols[cm]))(row);
        IPosition pos(2,4,iodata.NchanMS[cm]);
        //Array<Complex> data = dataCol(row);
        for(int k = 0; k < iodata.NchanMS[cm]; k++) {
           pos(0)=0;pos(1)=k;
           data(pos)=Complex(iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8],iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+1]);
           pos(0)=1;pos(1)=k;
           data(pos)=Complex(iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+2],iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+3]);
           pos(0)=2;pos(1)=k;
           data(pos)=Complex(iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+4],iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+5]);
           pos(0)=3;pos(1)=k;
           data(pos)=Complex(iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+6],iodata.xo[iodata.Nbase*iodata.tilesz*8*chanoff+row0*8+7]);
           chanoff++;
       }

       (*(dataCols[cm])).put(row,data);
       //dataCol.put(row,data); // copy to output

   }
       row0++;

      }

    }
    for (int cm=0; cm<iodata.Nms;cm++) {
     delete dataCols[cm];
    }

}

void 
Data::writeDataMinibatch(Table ti, Data::IOData iodata, int minibatch) {

    /* first iterate to the right minibatch */
    Block<String> ivl(1); ivl[0]="TIME";
    TableIterator tit(ti,ivl);
    /* till which timeslot should we iterate ? */
    int tillts=minibatch*iodata.tilesz;
    int ttime=0;
    while(!tit.pastEnd() && ttime<tillts) {
      tit.next();
      ttime++;
    }

    /* sort input table by ant1 and ant2 */
    Block<String> iv1(2);
    iv1[0] = "ANTENNA1";
    iv1[1] = "ANTENNA2";

    /* how many timeslots to read now, if we have reached a valid row */
    int tmb=0;
    int rowoffset=0;
    while(!tit.pastEnd() && tmb<iodata.tilesz) {


    Table t=tit.table().sort(iv1,Sort::Ascending);

    ROScalarColumn<int> a1(t, "ANTENNA1"), a2(t, "ANTENNA2");
    /* writable access for output */
    ArrayColumn<Complex> dataCol(t, Data::OutField);

    /* check we get correct rows = baselines+stations */
    int nrow=t.nrow(); 
    //cout<<"Table rows "<<nrow<<" Data rows "<<iodata.tilesz*iodata.Nbase+iodata.tilesz*iodata.N<<endl;
    int row0=rowoffset;
    IPosition pos(2,4,iodata.Nchan);
    for(int row = 0; row < nrow; row++) {
        uInt i = a1(row); //antenna1 
        uInt j = a2(row); //antenna2
        /* only work with cross correlations */
        if (i!=j) {
        Array<Complex> data = dataCol(row);
        for(int k = 0; k < iodata.Nchan; k++) {
           pos(0)=0;pos(1)=k;
           data(pos)=Complex(iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8],iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+1]);
           pos(0)=1;pos(1)=k;
           data(pos)=Complex(iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+2],iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+3]);
           pos(0)=2;pos(1)=k;
           data(pos)=Complex(iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+4],iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+5]);
           pos(0)=3;pos(1)=k;
           data(pos)=Complex(iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+6],iodata.xo[iodata.Nbase*iodata.tilesz*8*k+row0*8+7]);
       }

       row0++;
       dataCol.put(row,data); // copy to output
      }
    }
     tmb++;
     rowoffset =row0;
     /* go to next timeslot */
     tit.next();
    }

}



void Data::freeData(Data::IOData data)
{
   delete [] data.u;
   delete [] data.v;
   delete [] data.w;
   delete [] data.x;
   delete [] data.xo;
   delete [] data.freqs;
   delete [] data.flag;
   delete [] data.NchanMS;
}


void Data::freeData(Data::IOData data, Data::LBeam binfo)
{
   delete [] data.u;
   delete [] data.v;
   delete [] data.w;
   delete [] data.x;
   delete [] data.xo;
   delete [] data.freqs;
   delete [] data.flag;
   delete [] data.NchanMS;

   delete [] binfo.time_utc;
   delete [] binfo.Nelem;
   delete [] binfo.sx;
   delete [] binfo.sy;
   delete [] binfo.sz;
   for (int ci=0; ci<data.N; ci++) {
     delete [] binfo.xx[ci];
     delete [] binfo.yy[ci];
     delete [] binfo.zz[ci];
   }
   delete [] binfo.xx;
   delete [] binfo.yy;
   delete [] binfo.zz;
}
