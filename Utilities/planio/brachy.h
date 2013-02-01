
/* brachy.h */

#ifndef _PLUNC_HAVE_BRACHY_H
#define _PLUNC_HAVE_BRACHY_H

#define MG 1
#define MC 2

/*
  defines for lookup tables
*/
#define POLAR_RADII (50)
#define POLAR_ANGLES (46)
#define MAXIMUM_RADIUS (30.0)
#define SEED_RADII (3001)	/* up to and including 30.0 cm */

typedef struct
{
    char isotope[50];
    float gamma;	/*  R per (MG or MC) per hour at 1 cm  */
    int gammaUnits;	/*  MG or MC  */
    float R_to_r;	/*  rad/Roentgen factor  */
    float half_life;	/*  in hours  */
    int TA_count;	/*  number of entries in tissue atten. table  */
    float tissue_attenuation[20];
    float TA_distance[20];
    float mu;		/*  linear attenuation coefficient - used past
				end of TA table  */
    int last_entry;	/*  boolean  */

                        /*  The next table is for a precalculated
			    table of values on a radius vector.  The
			    corresponding radii are on .1 mm increments
			    such that the table index is the radius in
			    .1 millimeters.  The dose engine should use
			    nearest-neighbor look-up.  This should
			    result in some plan-time cycle savings
			    */
    float seed_dose_table[SEED_RADII];
} SEED_SPEC;

typedef struct
{
    char isotope[50];
    float gamma;	/*  R per (MG or MC) per hour at 1 cm  */
    int gammaUnits;	/*  MG or MC  */
    float R_to_r;	/*  rad/Roentgen factor  */
    float half_life;	/*  in hours  */
    float phys_length;	/*  physical length of source  */
    float act_length;	/*  active length of source - assumed to be
			    centered in physical length  */
    float wall;		/*  thickness of capsule wall  */
    float wall_mu;	/*  attenuation coeff. of capsule material  */
    float diameter;	/*  outside diameter of capsule  */
    float TA_fit[4];	/*  tissue attenuation fit ala Meisberger -
			  	constant term first  */
    float mu;		/*  linear tissue attenuation coefficient - used past
				10 cm  */
    int last_entry;	/*  boolean  */

                        /*  The next three data are for a precalculated
			    table of values on a polar grid.  This should
			    result in some plan-time cycle savings
			    */
    float polar_radii[POLAR_RADII];
    float polar_angles[POLAR_ANGLES];
    float polar_table[POLAR_ANGLES][POLAR_RADII];
} SOURCE_SPEC;

typedef struct
{
    int type;
    PNT3D p[2];		/*  entered coordinates in cm  */
} SOURCE;

typedef struct
{
    int type;
    PNT3D p;		/*  entered coordinates in cm  */
} SEED;

typedef struct
{
    char label[100];
    int seed_count;	/* number of seeds in implant */
    int source_count;	/* number of sources in implant */
    Boolean *seed_list;		/* is seed in object or no? */
    Boolean *source_list;	/* is source in object or no? */
} BRACHY_OBJECT;

typedef struct
{
    int count;
    BRACHY_OBJECT *objects;
} BRACHY_OBJECTS;

typedef struct
{
    int		source_count;
    int		seed_count;
    SOURCE	*sources;
    SEED	*seeds;
} IMPLANT_DESC;
/*
  When written to file, an IMPLANT_DESC is followed by the SOURCE and
  SEED arrays as indicated.
*/

#endif
