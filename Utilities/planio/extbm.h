
/* extbm.h */

/*
  This header file contains data type and record declarations specific
  to external beam calculations.
*/

#ifndef _PLUNC_HAVE_EXTBM_H
#define _PLUNC_HAVE_EXTBM_H

#define MAX_FILTER_COUNT	(2)	/* per beam */
#define MAX_CHUNK_COUNT		(4)

/*
  SAR table sizes
*/
#define MAX_SAR_DEPTHS		(30)
#define MAX_SAR_RADII		(30)
#define MAX_FLATNESS_DEPTHS	(5)
#define MAX_FLATNESS_RADII	(40)
#define MAX_Sc_POSITIONS	(25)	/* field sizes for output factors */

/*
  Some constants for dose engine
*/
#define dSAR_MAX_RADII		(30)
#define REASONABLE_RADIUS_COUNT	(15)
#define dSAR_MAX_ANGLES		(60)
#define REASONABLE_SECTOR_COUNT	(15)

/*  types of placement  */
#define SSD_PLACEMENT		(1)
#define ISOCENTRIC		(2)

/*  types of inhomogeneity correction  */
#define BATHO			(1)
#define EQ_TAR			(2)

/*  type of beam movement during treatment  */
#define STATIC			(1)
#define ARC			(2)

/*  types of treatment units  */
#define PHOTON			(1)
#define ELECTRON		(2)
#define SIMULATOR		(3)

/*  types of scatter tables  */
#define TAR			(1)
#define TPR			(2)
#define TMR			(3)

#define S_THING(type) (((type) == TAR) ? "SAR" : \
		       (((type) == TMR) ? "SMR" : \
			(((type) == TPR) ? "SPR" : \
			 "S*R")))
#define T_THING(type) (((type) == TAR) ? "TAR" : \
		       (((type) == TMR) ? "TMR" : \
			(((type) == TPR) ? "TPR" : \
			 "T*R")))

/*  types of treatment times  */
#define MU			(1)
#define ENGLISH			(2)
#define DECIMAL_UNIT		(3)

#define TIME_UNITS(units) (((units) == MU) ? "monitor unit(s)" : \
			   (((units) == ENGLISH) ? "minute(s)" : \
			    (((units) == DECIMAL_UNIT) ? "minute(s)" : \
			     "mystery unit(s)")))

#define MAX_JAW_COUNT		(50)
#define MAX_SEGMENT_COUNT	(20)
typedef struct
{
    int independent;		/* 0:symmetric  1:independent  2:multi-leaf */
/*
  c. axis is 0.0 and these are relative to that.  1 is left or bottom
  and 2 is right or top.
  Since these are signed quantities the sense of 1 is counterintuitive.
  For example, a typical traditional accelerator may have values:
	independent = 0
	min_1 = -20.0
	max_1 = -2.0
	min_2 = 2.0
	max_2 = 20.0
	min_w = -20.0;
	max_w = 20.0;
*/
    float min_1;	/* leftmost or bottommost travel */
    float max_1;	/* rightmost or topmost travel */
    float min_2;	/* leftmost or bottommost travel */
    float max_2;	/* rightmost or topmost travel */
    float min_w;	/* start of jaw orthogonal to jaw travel dir */
    float max_w;	/* end of jaw */
    float SDD;		/* primary source to jaw (diaphram) distance */
    char  name_1[NAME_LENGTH], name_2[NAME_LENGTH];
} JAW;

typedef struct
{
    int independent;
    float pos_1;	/* 1 is left or bottom, 2 is right or top */
    float pos_2;	/* These are jaw positions (0.0 is at c-axis) */
    float min_w;
    float max_w;
} JAW_POSITION;

typedef struct
{
    int version;
    int unit_id;	/* unique unit id number */
    char name[NAME_LENGTH];
    int modality;	/* PHOTON, ELECTRON, SIMULATOR  */
    float dmax;		/* the nominal depth of maximum dose */
    float SAD;		/* Source to Axis Distance */
    float PSD;		/* Primary src to Secondary src Distance */
    float source_diam1;	/* for penumbra calculations, primary  */
    float source_diam2;	/* for penumbra calculations, secondary  */
    float source_ratio;	/* fraction of source due to primary */

    int x_jaw_count;
    int y_jaw_count;
    JAW jaw[MAX_JAW_COUNT];

    int gantry_sense;	  /* +1 => +angle=ccw ; -1 => +angle=cw rotation */
    float gantry_offset;  /* currently unused */
    float gantry_minimum; /* minimum limit of rotation */
    float gantry_maximum; /* maximum limit of rotation */

    int table_sense;
    float table_offset;
    float table_minimum;
    float table_maximum;

    int collimator_sense;
    float collimator_offset;
    float collimator_minimum;
    float collimator_maximum;
} UNIT;

/*
 *  The time calculation parameters will normally be read from an
 *  ASCII file (.../phys_dat/time_calc ?) so it can be easily modified
 */
typedef struct
{
    int version;
    int unit_id;
    float dose_rate;	/* calibrated dose rate at cal_dis and cal_depth */
    PDATE calibration_date;
    float decay_constant;	/* inverse days */
    float cal_dist;	/* SSD + depth for calibration */
    float cal_depth;	/* depth of calibration; negative if "in air" */
    float end_effect;	/* actual time = set time + end_effect */
    int time_units;	/* one of MU, ENGLISH, DECIMAL */
/*
  Output factor - this corrects for dose rate variation due strictly
  to changes in collimator size.

  The numbers below are used in a way consistent with Khan's Sc.  The
  value for a given field is looked for the rectangular collimator
  setting without regard to any beam shaping.

  The output factors, or Sc values if you prefer, are tabulated for
  your choice of rectangular fields.

  *NOTE* that the Sc_*_positions below are the collimator position.  For
  instance, the x and y positions for a 10 by 10 would by 5 and 5.

  But what about independent jaws?  Gad, I hate to think...
*/
    int Sc_x_count;	/* number of positions in x */
    int Sc_y_count;	/* number of positions in y */
    float Sc_x_positions[MAX_Sc_POSITIONS];
    float Sc_y_positions[MAX_Sc_POSITIONS];
    float Sc[MAX_Sc_POSITIONS][MAX_Sc_POSITIONS];	/* Sc[y][x] */
} TIME_CALC_PARAMETERS;

#define REAL_FILTER		0
#define DYNAMIC_FILTER		1
#define BOLUS_FILTER		2
#define COMPENSATOR_FILTER	3
#define CUSTOM_FILTER_ID2	-3

/* Pre-version 6 style */
#define CUSTOM_FILTER_ID	-1
#define ANASTRUCT_FILTER_ID	-2

typedef struct
{
    int	version;
    int unit_id;
    int type;
    int filter_id;	/* unique filter id number */
/*
  Mirrorness is with respect to the y axis of the beam.  Conventional
  wedges which have their point toward or away from the gantry have
  themselves as mirrors.  Conventional wedges which have their point
  to the right or left usually have mirror images (often the same
  physical wedge inserted the other way around).  If a filter has a
  mirror wrt to the y axis, list the mirror's id here for use with
  copy and oppose. Otherwise, list NO_MIRROR.
*/
#define NO_MIRROR (0)
    int mirror_id;
    int next_id;
    int prev_id;
    char name[NAME_LENGTH];
#define NOT_POINTY (0)
#define NORTH (1)
#define SOUTH (2)
#define EAST (3)
#define WEST (4)
    int pointy_end;	/* Where is the pointy end? One of above. */
    JAW jaw_limits[2];	/* jaw limits for use of filter */
/*
  The filter is defined as a collection of prisms. I have made the arbitrary
  decision that prisms are defined as polygons in x-y and translated
  along z.  Hence for a wedge, z is perpendicular to the central axis.
  For a compensator, z is probably going to be parallel to the c. axis.
  That's what these two transforms are for - to move between the filter
  definition coordinate system and the beam (B) system.  Don't forget
  that B is left-handed.  And don't forget to translate to the proper
  source distance.
*/
    float T_filter_to_beam[4][4];
    float T_beam_to_filter[4][4];
    float mu[MAX_CHUNK_COUNT];
    float mu_dx[MAX_CHUNK_COUNT];
    float mu_dr[MAX_CHUNK_COUNT];
    float hvl_slope[MAX_CHUNK_COUNT];
    /*
      Store the linear attenuation coefficient (inverse cm, positive)
      for each prism in the prism's contours[foo].density spot.
      The min and max entries of the ANASTRUCT and CONTOURs are ignored
      except for the CONTOUR's min.z and max.z.
    */
    /* For virtual wedge wedge-factor */
    float wfo_ao;
    float dwfo_da;
    float dwf_dfs_ao;
    float dwf_dfs_da;
    ANASTRUCT profiles;
} FILTER;

#define SQUARE_HOLE	(1)
#define ROUND_HOLE	(2)
#define SQUARE_SLOT	(3)
#define ROUND_SLOT	(4)
#define MLC_HOLE	(5)
typedef struct
{
    int shape;		/* SQUARE_HOLE, ROUND_HOLE, etc. */
    float x;		/* center of hole and inside dimensions */
    float y;
    float height;
    float width;	/* ignored for SQUARE and ROUND */
} TRAY_HOLE;

typedef struct
{
    int	version;
    int unit_id;	/* unit on which it can be used */
    int tray_id;	/* unique ID number */
    char name[NAME_LENGTH];
    float tray_factor;	/* for this unit */
    float xmin;		/* tray size - can be asymmetric */
    float xmax;
    float ymin;
    float ymax;
    float tray_dist;	/*  for block templates */
    float block_dist;	/*  bottom of block for penumbra calc -
			    different from tray_dist so block can hang
			    from tray. */
    int hole_count;	/* can be 0 - must be set */
    TRAY_HOLE *hole_list;	/* courtesy of malloc */
} TRAY;

typedef struct
{
    int	version;
    int unit_id;
    int type;		/* one of TAR, TPR, TMR */

    float tran;		/* collimator transmission factor */
/*
  table parameters and tables
*/
    int depth_count;
    int radius_count;
    int flatness_depth_count;
    int flatness_radius_count;	/*  number of active slots in flatness table */
    float depth_vec[MAX_SAR_DEPTHS];
    float radius_vec[MAX_SAR_RADII];
    float flatness_depth_vec[MAX_FLATNESS_DEPTHS];
    float flatness_radius_vec[MAX_FLATNESS_RADII];
    float SAR[MAX_SAR_DEPTHS][MAX_SAR_RADII];
    float TAR0[MAX_SAR_DEPTHS];
/*
  One to MAX_FLATNESS_DEPTHS flatness profiles may be included here.
  The radius vector is at machine SAD and the flatness profiles are
  assumed to be depth scaled accordingly.
*/
    float flatness[MAX_FLATNESS_RADII][MAX_FLATNESS_DEPTHS];
/*
  This is a Phantom Scatter Factor (in principle the same as Khan's
  Sp) which is intended to correct for the change in scatter dose to
  the reference depth as a function of irradiated phantom area.  We do
  a Clarkson-type summation to determine the Sp for any given point.
  This table uses the same radii as the SAR table.  It is ignored for
  SAR/TAR calculations.
*/
    float Sp[MAX_SAR_RADII];
/*
  As delivered the Sp table is not suitable for
  the kind of Clarkson summation we intend to do with it,
  primarily because Sp(0) != 0.0  so what we do is to subtract
  the Sp(0) value from the table and thus build an integratable
  table stored here.  Then we just do a Clarkson-type integration
  and then add back the Sp0.
*/
    float Sp0;
    float Sp_prime[MAX_SAR_RADII];
} SAR_TABLE;

typedef struct
{
    unsigned long serial_number;	/* unique instance id */
    char name[NAME_LENGTH];
    int unit_id;
    int x_jaw_count;
    int y_jaw_count;
    int segment_count;
    float *segment_weight;
    JAW_POSITION **jaw;	/* collimator setting: array of segment ptr's to JAW_POS's  */
/*
  The following are for people to use...
*/
    PNT3D position;		/*  patient coordinates of central axis at
				    unit's definition distance */
    PNT3D centroid;	/* Center of rotation, = position for ISOCENTRIC */
    float gantry_angle;	/*  "standard machine" reading for this beam */
    float coll_angle;	/*  "standard machine" reading for this beam */
    float table_angle;	/*  "standard machine" reading for this beam */

    float SSD;		/* provided by virtual simulator */
/*
  ...but this transform matrix is all the computer needs to know.
*/
    float T_pat_to_beam[4][4];	/*  transformation matrix that takes patient
				    coordinates to beam coordinates */
    float T_beam_to_pat[4][4];  /* and the inverse */

    int placement;	/*  SSD_PLACEMENT or ISOCENTRIC */
    int extent;		/*  STATIC, ARC  */

#define INH_CORR	(0x1)
#define CON_CORR	(0x2)
    int corrections;	/* bit OR of INH_CORR, CON_CORR, etc. */
    int inh_type;	/*  BATHO, EQ_TAR  */

#define NO_FILTER	(-1)
    int filter_id[MAX_FILTER_COUNT];
    int	filter_type[MAX_FILTER_COUNT];
    int vw_angle;

#define NO_TRAY		(-1)
    int tray_id;	/* NO_TRAY if none */

/*
  The following three entries are of interest only for gantry-only
  arcs.  This section will get generalized later if the need arises to
  do compound-motion arcs.
*/
    float start_angle;
    float stop_angle;
    float angle_inc;

/*
  This is the beam outline at the SAD for the treatment unit. A
  vertex_count of 0 indicates no custom block. The z value of the
  CONTOUR must be set to the SAD for the treatment machine.
*/ 
    CONTOUR beam_outline;

/*
  This is the collimated version of the same outline.  It has a
  vertex_count > 2 and represents the intersection of the beam_outline
  with the collimator setting.
*/
    CONTOUR collimated_beam_outline;

/*
  A custom filter (such as bolus or compensator) needs to have a set
  of profiles describing its shape, note, all general information
  about it is in the FILTER file.
*/

    ANASTRUCT	custom_filter[2];

} EXT_BEAM;

#endif

