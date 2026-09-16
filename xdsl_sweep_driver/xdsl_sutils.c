/* 
   GEMM FLAVOURS

   -----

   GEMM FLAVOURS is a family of algorithms for matrix multiplication based
   on the BLIS approach for this operation: https://github.com/flame/blis

   -----

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
   You should have received a copy of the GNU General Public License along with
   this program. If not, see <http://www.gnu.org/licenses/>.

   -----

   author    = "Enrique S. Quintana-Orti"
   contact   = "quintana@disca.upv.es"
   copyright = "Copyright 2021, Universitat Politecnica de Valencia"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/

#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]
#define Mrow(a1,a2)  M[ (a1)*(ldM)+(a2) ]

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "dtypes.h"

void generate_matrix( char orderM, size_t m, size_t n, void *vM, size_t ldM, int type_id )
{
/*
 * Generate a matrix with random entries
 * m      : Row dimension
 * n      : Column dimension
 * vM     : Matrix (void* to handle different types)
 * ldM    : Leading dimension
 * type_id: 0 for A/B (A_TYPE/B_TYPE), 1 for C (C_TYPE)
 */
  int i, j;
  
  if ( orderM=='C' ) {
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ ) {
        if (type_id == 0) {
            A_TYPE *M = (A_TYPE*)vM;
#if defined(FP16)
            Mcol(i,j) = ((((A_TYPE) j*m+i)/m)/n);
#elif defined(I8I8I32)
            Mcol(i,j) = (A_TYPE)((i+j)%5 + 1);
#else
            Mcol(i,j) = ((A_TYPE) rand())/RAND_MAX + 1.0;
#endif
        } else {
            C_TYPE *M = (C_TYPE*)vM;
#if defined(FP16)
            Mcol(i,j) = ((((C_TYPE) j*m+i)/m)/n);
#elif defined(I8I8I32)
            Mcol(i,j) = (C_TYPE)123;
#else
            Mcol(i,j) = ((C_TYPE) rand())/RAND_MAX + 1.0;
#endif
        }
      }
  }else
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ ) {
        if (type_id == 0) {
            A_TYPE *M = (A_TYPE*)vM;
#if defined(FP16)
            Mrow(i,j) = ((((A_TYPE) j*m+i)/m)/n);
#elif defined(I8I8I32)
            Mrow(i,j) = (A_TYPE)((i+j)%5 + 1);
#else
            Mrow(i,j) = ((A_TYPE) rand())/RAND_MAX + 1.0;
#endif
        } else {
            C_TYPE *M = (C_TYPE*)vM;
#if defined(FP16)
            Mrow(i,j) = ((((C_TYPE) j*m+i)/m)/n);
#elif defined(I8I8I32)
            Mrow(i,j) = (C_TYPE)123;
#else
            Mrow(i,j) = ((C_TYPE) rand())/RAND_MAX + 1.0;
#endif
        }
      }
}
/*===========================================================================*/
void print_matrix( char *name, char orderM, size_t m, size_t n, void *vM, size_t ldM, int type_id )
{
/*
 * Print a matrix to standard output
 * name   : Label for matrix name
 * m      : Row dimension
 * n      : Column dimension
 * vM     : Matrix
 * type_id: 0 for A/B, 1 for C
 */
  size_t i, j;

  if ( orderM=='C' )
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ ) {
        if (type_id == 0) {
            A_TYPE *M = (A_TYPE*)vM;
#if defined(I8I8I32)
            printf( "%s[%zu,%zu] = %d;\n", name, i, j, (int) Mcol(i,j) );
#else
            printf( "%s[%zu,%zu] = %f;\n", name, i, j, ((double) Mcol(i,j)) );
#endif
        } else {
            C_TYPE *M = (C_TYPE*)vM;
#if defined(I8I8I32)
            printf( "%s[%zu,%zu] = %d;\n", name, i, j, (int) Mcol(i,j) );
#else
            printf( "%s[%zu,%zu] = %f;\n", name, i, j, ((double) Mcol(i,j)) );
#endif
        }
      }
  else
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ ) {
        if (type_id == 0) {
            A_TYPE *M = (A_TYPE*)vM;
#if defined(I8I8I32)
            printf( "%s[%zu,%zu] = %d;\n", name, i, j, (int) Mrow(i,j) );
#else
            printf( "%s[%zu,%zu] = %f;\n", name, i, j, ((double) Mrow(i,j)) );
#endif
        } else {
            C_TYPE *M = (C_TYPE*)vM;
#if defined(I8I8I32)
            printf( "%s[%zu,%zu] = %d;\n", name, i, j, (int) Mrow(i,j) );
#else
            printf( "%s[%zu,%zu] = %f;\n", name, i, j, ((double) Mrow(i,j)) );
#endif
        }
      }
}
/*===========================================================================*/
double dclock()
{
/* 
 * Timer
 *
 */
  struct timeval  tv;
  // struct timezone tz;

  gettimeofday( &tv, NULL );   

  return (double) (tv.tv_sec + tv.tv_usec*1.0e-6);
}
/*===========================================================================*/
