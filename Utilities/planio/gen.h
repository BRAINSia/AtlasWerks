/*
  This header file contains data type and record declarations of general use
  in the planning system.
*/

#ifndef _PLUNC_HAVE_GEN_H
#define _PLUNC_HAVE_GEN_H

#ifdef PLAN_WINNT
//#include "libmisc.h"
#endif

#ifndef VAR_GREYBAR_ENTRIES
#define GREYBAR_ENTRIES	(100)
#endif

/* This version number is always 3 numeric digits and ties the struct's
 * members together so that all that needs to be known is this number.
 * When any struct in src/include changes the PLUNC version number
 * MUST be changed as well. 
 * It is meant to be compared using < or > to hardcoded values, as in
 *   #if PLUNC_VERSION > 322
 */
#define PLUNC_VERSION 420	/* means 4.2.0 */


#ifndef TRUE
#define TRUE (1)
#endif
#ifndef FALSE
#define FALSE (0)
#endif
#ifndef PI
#define PI (3.141592654)
#endif

/* Fixed point math scaling values */
#define SBITS		14
#define SCALE		16384
#define INV_SCALE	6.1035156e-5
#define MASK		0x3fff

/*
  Some convenience macros
*/
#define PLAN_MIN(a,b)		(((a) < (b)) ? (a) : (b))
#define PLAN_MAX(a,b)		(((a) > (b)) ? (a) : (b))
#define PLAN_ABS(a)		(((a) < 0) ? -(a) : (a))
#define PLAN_FABS(a)		(((a) < 0.0) ? -(a) : (a))
#define PLAN_CLOSE_TO(a,b,t)	(PLAN_ABS((a) - (b)) <= (t))
#define PLAN_NITEMS(array)      (sizeof(array) / sizeof(*(array)))

/* limit a radian angle to (-90..+90) */
#define LIMIT90(ANGLE) \
      if ((ANGLE) >=  PI/2.) (ANGLE) = (ANGLE) - PI; \
      if ((ANGLE) <= -PI/2.) (ANGLE) = (ANGLE) + PI;

#define FI_LIMIT		(40)
#define TABLET			(0)
#define KEYBOARD		(1)

#define CT_TO_ENCODED		(1024)


/*
  Everything's got a name.
*/
#define NAME_LENGTH		(100)

typedef int Boolean;
#ifndef Bool
#define Bool int
#endif

#ifndef _PLUNC_HAVE_LIB3D_H
typedef struct
{
    float x;
    float y;
    float z;
} PNT3D;


typedef float LIB3D_TRANSFORM[4][4];
#endif

typedef struct
{
    int		count;
    int		con_num1;
    int		con_num2;
    int		*vert;
    PNT3D	*pt;
    PNT3D	*norm;
} T_STRIP;

typedef struct
{
    int		count;
    T_STRIP	*strip;
} T_STRIPS;

typedef float COLOR4[4];

typedef struct
{
    int day; /* one based: 1 = first day of month */
    int month; /* one based: 1 = Jan */
    int year; /* 1900 based: 97 = 1997 (year 2000 problem) */
    int dow; /* zero based: 0 = Sunday */
} PDATE;

typedef struct
{
    int factors;	/* weights can be products of several */
			/* components - for instance a brachytherapy */
			/* object can be weighted by a time and an */
			/* activity (and maybe a decay factor) whereas */
			/* a beam weight is a beam renormalization */
			/* factor times a relative weight factor */
    union
    {
	float wt[5];
	struct
	{
	    float time;
	    float activity;
	    float decay;
	} brachy;
	struct
	{
	    float renorm;		/* for renormalizing beam */
	    float relative;		/* for combination with others */
	    float plan_norm;		/*
					  yet another multiplier,
					  embodying target dose and
					  target percent
					  */
	    float fractions;
	} tele;
    }f;
} WEIGHT;

typedef struct
{
    int count;
    WEIGHT *weights;
} WEIGHTS;
/*
  A weight file consists of a WEIGHTS structure followed by an array
  of WEIGHT.
*/

/*******************************************
  The 2D grid is a thing of the past.  It may come in handy for
  producing isodoses on slice planes and so is temporarily retained
  here.
*/
typedef struct
{
    float x_start;
    float y_start;
    float x_inc;
    float y_inc;
    int x_count;
    int y_count;
    float grid_to_pat_T[4][4];
    float pat_to_grid_T[4][4];
    float max;
    float min;
    float *matrix;
    Boolean *inside;
} TWOD_GRID;
#define TWOD_GRID_VALUE(GRID_ptr, i, j) ((GRID_ptr)->matrix[(i) +\
			((j) * (GRID_ptr)->x_count)])
/*
  A 2D dose grid file is prefaced by a TWOD_GRID and includes:

  float dose_matrix[y_count][x_count];
  Boolean inside[y_count][x_count];
*/

typedef struct
{
    PNT3D start;
    PNT3D inc;
    int x_count;
    int y_count;
    int z_count;
    float grid_to_pat_T[4][4];
    float pat_to_grid_T[4][4];
    float max;
    float min;
    float *matrix;
} GRID;
#define GRID_VALUE(GRID_ptr, i, j, k)\
    ((GRID_ptr)->matrix[(i) +\
			(GRID_ptr)->x_count *\
			 ((j) + ((k) * (GRID_ptr)->y_count))])
/*
  A 3D dose grid file is prefaced by a GRID and includes:

  float dose_matrix[z_count][y_count][x_count];
*/

/*
  The CALC_POINT is just a special case of the dose GRID.
*/
typedef struct
{
    char label[NAME_LENGTH];
    PNT3D p;
} CALC_POINT;

typedef struct
{
    int count;
    CALC_POINT *points;
    float *dose;
} CALC_POINTS, POINT_DOSES;
/*
  The point file contains a CALC_POINTS followed by an array of
  CALC_POINT.
*/
/*
  The point *dose* file contains a CALC_POINTS followed by an array of
  CALC_POINT followed by the point doses (float doses[count]).
*/

/* GRIDS and C_POINTS are used in both photon and electron dose code: */
typedef struct
{
    GRID	grid3;
    int		out_fdes;
} GRIDS;

typedef struct
{
    CALC_POINTS points;
    float	*ssd;
    float	*batho;
    int		out_fdes;
} C_POINTS;

/*******************************************
  And then there's contours
*/
typedef struct
{
    int vertex_count;
    int slice_number;		/* number of associated CT slice */
    PNT3D max;			/* extent of prism represented by this */
				/* contour */
    PNT3D min;
    float density;		/* rayline thickness multiplier */
    float z;			/* nominal slice position  */
    float *x;
    float *y;
} CONTOUR;
/*
  A 2D contour file is prefaced by a CONTOUR and includes:

  float vertices[2][vertex_count];
*/

/*
  There is one ANASTRUCT per structure.  Contours are not
  constrained to be in separate planes so n-furcation is explicitly
  supported.
*/
typedef struct
{
    char label[NAME_LENGTH];
    int contour_count;
    PNT3D max;
    PNT3D min;
    CONTOUR *contours;
} ANASTRUCT;
/*
  An anastruct file is prefaced by an ANASTRUCT and is the
  concatenation of contour_count CONTOUR files (see above)
*/

/**********************************
  Plan description - this will get *much* more complicated.
*/
typedef struct
{
    float pat_to_plan_T[4][4];	/* if 2D, z=0.0 plane is selected, if */
				/* 3D this is just a view */
    int isodose_count;
    float ref_dose;		/* The isodose values are % of this ref dose */
    float *isodoses;
} PLAN;
/*
  plan file consists of PLAN structure followed by vector of isodoses
*/

#ifndef DCTYPES_H
// SBYTE declared the same way in both DCMTK and CompOnc
//typedef signed char SBYTE;
#endif

typedef unsigned char UBYTE;

typedef struct
{
   int		x_dim;
   float	x_start,x_end;
   float	x_scale,inv_x_scale;
   int		y_dim;
   float	y_start,y_end;
   float	y_scale,inv_y_scale;
   int		z_dim;
   float	z_start,z_end;
   float	z_scale,inv_z_scale;
   int		slicecnt;
   float	*data;
   /* Only used for variable zdim map */
   int		depth_inc;
   UBYTE	*cnt_map;
   UBYTE	*max_map;
   float	**vdata;
} THREED_MAP;

typedef struct
{
    int		num;
    int		current;
    int		*point;
    int		*points;
    int		*beam;
    float	*dose;
    float	*fraction;
    float	*percent;
    char	**label;
} NORM;

typedef struct
{
    float	val;
    float	weight;
} DOSE_INDEX;

/*
typedef struct {
    int		num_entries;
    float	bin_size;
    float	*vol;
} INDEX_LIST;
*/

typedef struct {
    int		target;
    int		type;
    float	weight;
    int		count;
    float	*dose;
    float	*vol;
    float	goal;
    float	total;
} USER_DVH;

typedef struct {
    int		num_entries;
    float	bin_size;
    float	*vol;
} DVH;

typedef struct
{
    char	name[NAME_LENGTH];
    int		numbeams;
    PNT3D	*beam_vectors;
} TEMPLATE;

typedef struct
{
    unsigned char	red;
    unsigned char	green;
    unsigned char	blue;
} RGB_PIXEL;

typedef struct
{
    int		tolerance_photon;
    int		tolerance_electron;
} LANTIS_DATA;

#endif
