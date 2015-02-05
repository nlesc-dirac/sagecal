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
#include <measures/Measures/MDirection.h>
#include <measures/Measures/UVWMachine.h>
#include <casa/Quanta.h>
#include <casa/Quanta/Quantum.h>

/* speed of light */
#ifndef CONST_C
#define CONST_C 299792458.0
#endif

using namespace casa;

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
int Data::format=0; /* old LSM */
double Data::nulow=2.0;
double Data::nuhigh=30.0;

int Data::max_emiter=3;
int Data::max_iter=2;
int Data::max_lbfgs=10;
int Data::lbfgs_m=7;
int Data::gpu_threads=128;
int Data::linsolv=1;
int Data::randomize=1;
int Data::DoSim=0;
int Data::DoDiag=0;
int Data::doChan=0; /* if 1, solve for each channel in multi channel data */
int Data::solver_mode=0;
int Data::ccid=-99999;
double Data::rho=1e-9;
char *Data::solfile=NULL;
char *Data::ignorefile=NULL;
char *Data::MSlist=NULL;

/* distributed sagecal parameters */
int Data::Nadmm=1;
int Data::Npoly=2;
double Data::admm_rho=5.0;

/* no upper limit, solve for all timeslots */
int Data::Nmaxtime=0;

using namespace Data;

void
Data::readMSlist(char *fname, vector<string> *msnames) {
  cout<<"Reading "<<Data::MSlist<<endl;
     /* multiple MS */
     ifstream infile(fname);
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

    //obtain the chanel freq information
    //sprintf(buff, "%s/SPECTRAL_WINDOW", fname);
    //Table _freq = Table(buff);
    Table _freq = Table(_t.keywordSet().asTable("SPECTRAL_WINDOW"));
    ROArrayColumn<double> chan_freq(_freq, "CHAN_FREQ"); 
    data->Nchan=chan_freq.shape(0)[0];
    data->Nms=1;
   /* allocate memory */
   data->u=new double[data->Nbase*data->tilesz];
   data->v=new double[data->Nbase*data->tilesz];
   data->w=new double[data->Nbase*data->tilesz];
   data->x=new double[8*data->Nbase*data->tilesz];
   data->xo=new double[8*data->Nbase*data->tilesz*data->Nchan];
   data->freqs=new double[data->Nchan];
   data->flag=new double[data->Nbase*data->tilesz];
   data->NchanMS=new int[data->Nms];
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
  x: data to write size Nbase*8*tileze x 1
  ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
*/
void 
Data::loadData(Table ti, Data::IOData iodata) {

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
    if(nrow-iodata.N*iodata.tilesz>iodata.tilesz*iodata.Nbase) {
      cout<<"Error in rows"<<endl;
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
    for(int row = 0; row < nrow; row++) {
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
           if (!flag.data()[k]){
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
         cxx/=(double)nflag;
         cxy/=(double)nflag;
         cyx/=(double)nflag;
         cyy/=(double)nflag;
         if (dotaper) {
          cxx*=uvtaper;
          cxy*=uvtaper;
          cyx*=uvtaper;
          cyy*=uvtaper;
         }
         iodata.flag[row0]=0;
        } else {
         if (!nflag) {
         /* all channels flagged, flag this row */
          iodata.flag[row0]=1;
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
}

/* each time this is called read in data from MS, and format them as
  u,v,w: u,v,w coordinates (wavelengths) size Nbase*tilesz x 1 
  u,v,w are ordered with baselines, timeslots
  x: data to write size Nbase*8*tileze x 1
  ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
*/
void 
Data::loadDataList(vector<MSIter*> msitr, Data::IOData iodata) {
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
           if (!flag.data()[k]){
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
         cxx/=(double)nflag;
         cxy/=(double)nflag;
         cyx/=(double)nflag;
         cyy/=(double)nflag;
         if (dotaper) {
          cxx*=uvtaper;
          cxy*=uvtaper;
          cyx*=uvtaper;
          cyy*=uvtaper;
         }
         iodata.flag[row0]=0;
        } else {
         if (!nflag) {
         /* all channels flagged, flag this row */
          iodata.flag[row0]=1;
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
      cout<<"Error in rows"<<endl;
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
