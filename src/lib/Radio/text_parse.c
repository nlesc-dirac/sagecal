/*
 *
 Copyright (C) 2006-2026 Sarod Yatawatta <sarod@users.sf.net>  
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


#include "Dirac_radio.h"
#include <ctype.h>

/* skips comment lines */
int
skip_lines(FILE *fin)
{

  int c;
  do {
  if ( ( c = getc(fin) ) == EOF )
    return(-1);
  /* handle empty lines */
  if ( c == '\n' )
  continue; /* next line */
  if ( (c != '#') ) {
  ungetc(c,fin);
  return(0);
  } else { /* skip this line */
  do {
  if ( ( c = getc(fin) ) == EOF )
  return(-1);
  } while (  c != '\n') ;
  }
  } while( 1 );
}

/* skips rest of line */
int
skip_restof_line(FILE *fin)
{
  int c;
  do {
  if ( ( c = getc(fin) ) == EOF )
  return(-1);
  } while (  c != '\n') ;
  return(1);
}


/* reads the next string (isalphanumeric() contiguous set of characters)
  separated by spaces, tabs or a newline. If the last character read is newline
  1 is returned, else 0 returned. */
/* buffer is automatically adjusted is length is not enough */
int
read_next_string(char **buff, int *buff_len, FILE *infd) {
   int k,c,flag;
   k = 0;
   /* intialize buffer */
   (*buff)[0]='\0';
   /* skip leading white space */
   do {
   c=fgetc(infd);
   /* also handle DOS end of line \r\n */
   if(c=='\n' || c=='\r' || c==EOF) return 1;
   } while(c != EOF && isblank(c));
   if(c=='\n' || c=='\r' || c==EOF) return 1;
   /* now we have read a non whitespace character */
   (*buff)[k++]=c;
  if (k==*buff_len) {
    /* now we have run out of buffer */
    *buff_len += 30;
    if ((*buff = (char*)realloc((void*)(*buff),sizeof(char)*(size_t)(*buff_len)))==NULL) {
     fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
     exit(1);
    }
  }
   flag=0;
   while ( ((c = fgetc(infd)) != EOF ) && k < *buff_len) {
      if ( c == '\n' || c=='\r' ) {  flag=1; break; }
      if ( isblank(c) ) {  break; }/* not end of line */
      (*buff)[k++] = c;
      if (k==*buff_len) {
       /* now we have run out of buffer */
       *buff_len += 30;
       if((*buff = (char*)realloc((void*)(*buff),sizeof(char)*(size_t)(*buff_len)))==NULL) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
       }
      }
  }
  /* now c == blank , \n or EOF */
  if (k==*buff_len-1) {
    /* now we have run out of buffer */
    *buff_len += 2;
    if((*buff = (char*)realloc((void*)(*buff),sizeof(char)*(size_t)(*buff_len)))==NULL) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
   }
  }

  /* add '\0' to end */
  (*buff)[k++]='\0';
  return flag;
}
