/*
FILE : plan_xdr_defs.c ************************************************

PURPOSE
Low level plan data structure XDR definitions.

COMMENTS
Whenever a new data structure is defined and an i/o routine is to be written
for it start by defining an XDR primitive for it.
*/

#include <stdio.h>
#include <stdlib.h>
#include "plan_strings.h"
#include "plan_file_io.h"
#include "plan_xdr.h"

#include "gen.h"
#include "extbm.h"
#include "brachy.h"
#include "plan_im.h"

typedef struct
{ unsigned contour : 1;
  unsigned inh : 1;
} corrections;
 
#include "libplanio.h"

/*
 ---------------------------------------------------------------------------
 
   NAME

   plan_xdr_defs.c
 
 
   DESCRIPTION
 
   low level plan data structure XDR definitions. whenever a new data
   structure is defined and an i/o routine is to be written for it
   start by defining an XDR primitive for it.
 
   RETURN VALUE

   all return 0 on successful completion; otherwise -1;
 
 
   AUTHOR
 	J Thorn
 	Radiation Oncology
 	North Carolina Memorial Hospital
 	University of North Carolina
 	27 September 1988
 
 
 
 ---------------------------------------------------------------------------
*/


/*---------- ANASTRUCT ----------*/

/*FUNCTION: xdr_ANASTRUCT **********************************************

PURPOSE
XDR primitvie for ANASTRUCT

****************************************************************************/
int
xdr_ANASTRUCT(XDR *xdr, ANASTRUCT *an)
{
    char *cp;

    cp = &(an->label[0]);
    if (xdr_string(xdr, &cp, 100) &&
	xdr_int(xdr, &an->contour_count) &&
	xdr_POINT(xdr, &an->max) &&
	xdr_POINT(xdr, &an->min))
	return(TRUE);
    else
	return(FALSE);
}

/*----------  CONTOUR ----------*/
int
xdr_CONTOUR(XDR *xdr, CONTOUR *contour)
{
    if (xdr_int(xdr, &contour->vertex_count) &&
	xdr_int(xdr, &contour->slice_number) &&
	xdr_POINT(xdr, &(contour->max)) &&
	xdr_POINT(xdr, &(contour->min)) &&
	xdr_float(xdr, &(contour->density)) &&
	xdr_float(xdr, &(contour->z)))
	return(TRUE);
    else
	return(FALSE);
}


/*FUNCTION: xdr_CONTOUR_X_Y *************************************************

PURPOSE
XDR primitive for CONTOUR_X_Y

***************************************************************************/
int
xdr_CONTOUR_X_Y(XDR *xdr, float *xy, int count)
{
    /* dumps the floats in the CONTOUR x and y slots */

    int x;

    for (x=0; x<count; x++)
	if (! xdr_float(xdr, (xy + x)))
	    return(FALSE);

    return(TRUE);
}


/*----------  PNT3D ----------*/

/*FUNCTION: xdr_POINT *************************************************

PURPOSE
XDR primitive for PNT3D

***************************************************************************/
int
xdr_POINT(XDR *xdr, PNT3D *point)
{
    if (xdr_float(xdr, &point->x) &&
	xdr_float(xdr, &point->y) &&
	xdr_float(xdr, &point->z))
	return(TRUE);
    else
	return(FALSE);
}
/* ---------- Boolean ---------- */

/*FUNCTION: xdr_Boolean *************************************************

PURPOSE
XDR primitive for Boolean

***************************************************************************/
int
xdr_Boolean( XDR *xdr, Boolean *b)
{
    if (xdr_int(xdr, b))
	return(TRUE);
    else
	return(FALSE);
}


/*FUNCTION: xdr_corrections *************************************************

PURPOSE
XDR primitive for corrections

***************************************************************************/
int
xdr_corrections(XDR *xdr, corrections *c)
{
    /* these are bit fields - this may not be quite right */

    if (xdr_int(xdr, (Boolean *)c))
	return(TRUE);

    return(FALSE);
}


/*----------  EXT_BEAM ----------*/

/*FUNCTION: xdr_EXT_BEAM *************************************************

PURPOSE
XDR primitive for EXT_BEAM

***************************************************************************/
int
xdr_EXT_BEAM(XDR *xdr, EXT_BEAM *eb, int *version)
{
    char		*cp;
    unsigned int	 lp;
    int			i, j;

    cp = &(eb->name[0]);

    /*
    if (xdr->x_op == XDR_ENCODE &&
	eb->x_jaw_count == 1 && eb->y_jaw_count == 1)
	*version = eb->unit_id;
    else if (xdr->x_op == XDR_ENCODE &&
	     eb->segment_count == 1)
	*version = -1;
    else *version = -6;
    */
    *version = -6;
    bool_t serial_result;
#if defined(__APPLE__) && defined(__LP64__)
    {
    unsigned int serial;
    serial_result = xdr_u_long(xdr, &serial);
    eb->serial_number = serial;
    }
#else
    serial_result = xdr_u_long(xdr,&eb->serial_number);
#endif
    if(!(serial_result &&
	  xdr_string(xdr, &cp, NAME_LENGTH) &&
	  xdr_int(xdr, version)))
	return(FALSE);
    if (*version <= -2) {
	/* MLC with multiple segments has -2 and then unit_id */
	if (!xdr_int(xdr, &eb->unit_id))
	    return(FALSE);
	if (!(xdr_int(xdr, &eb->x_jaw_count) &&
	      xdr_int(xdr, &eb->y_jaw_count)))
	    return(FALSE);
	if (!(xdr_int(xdr, &eb->segment_count)))
	    return(FALSE);
    }
    else if (*version == -1) {
	/* MLC has -1 and then unit_id */
	if (!xdr_int(xdr, &eb->unit_id))
	    return(FALSE);
	if (!(xdr_int(xdr, &eb->x_jaw_count) &&
	      xdr_int(xdr, &eb->y_jaw_count)))
	    return(FALSE);
	eb->segment_count = 1;
    }
    else {
	/* Reading old style with X and Y jaws */
	eb->unit_id = *version;
	eb->x_jaw_count = 1;
	eb->y_jaw_count = 1;
	eb->segment_count = 1;
    }

    if (xdr->x_op == XDR_DECODE) {
	eb->segment_weight = (float *)
	    malloc(eb->segment_count*sizeof(float));
	eb->jaw = (JAW_POSITION **)
	    malloc(eb->segment_count*sizeof(JAW_POSITION *));
    }

    lp = eb->x_jaw_count + eb->y_jaw_count;

    for (j = 0; j < eb->segment_count; j++) {
      if (xdr->x_op == XDR_DECODE)
	eb->jaw[j] = (JAW_POSITION *)
	    malloc(MAX_JAW_COUNT*sizeof(JAW_POSITION));
      for(i = 0; i < static_cast<int>(lp); i++) {
	JAW_POSITION *jaw = eb->jaw[j] + i;
	if (*version <= -2) {
	    if (!(xdr_float(xdr, &(eb->segment_weight[j]))))
		return(FALSE);
	}
	else eb->segment_weight[j] = 1.0;
	if (*version <= -1) {
	    if (!(xdr_int(xdr, &(jaw->independent)) &&
		  xdr_float(xdr, &(jaw->pos_1)) &&
		  xdr_float(xdr, &(jaw->pos_2)) &&
		  xdr_float(xdr, &(jaw->min_w)) &&
		  xdr_float(xdr, &(jaw->max_w))))
		return(FALSE);
	}
	else {
	    if (!(xdr_int(xdr, &(jaw->independent)) &&
		  xdr_float(xdr, &(jaw->pos_1)) &&
		  xdr_float(xdr, &(jaw->pos_2))))
		return(FALSE);
	    jaw->min_w = -20.0;
	    jaw->max_w = 20.0;
	}
      }
    }

    if (!xdr_POINT(xdr, &eb->position)) return(FALSE);

    if (!(xdr_float(xdr, &eb->gantry_angle) &&
	  xdr_float(xdr, &eb->coll_angle) &&
	  xdr_float(xdr, &eb->table_angle) &&
	  xdr_float(xdr, &eb->SSD)))
	return(FALSE);

    for(i=0; i<4; i++)
	for(j=0; j<4; j++)
	    if (! xdr_float(xdr, &(eb->T_pat_to_beam[i][j])))
		return(FALSE);

    for(i=0; i<4; i++)
	for(j=0; j<4; j++)
	    if (! xdr_float(xdr, &(eb->T_beam_to_pat[i][j])))
		return(FALSE);

    if (!
	(xdr_int(xdr, &eb->placement) &&
	 xdr_int(xdr, &eb->extent) &&
	 xdr_int(xdr, &eb->corrections) &&
	 xdr_int(xdr, &eb->inh_type)))
	return(FALSE);

    if (*version == -6 && eb->placement == SSD_PLACEMENT) {
	if (!xdr_POINT(xdr, &eb->centroid)) return(FALSE);
    }
    else eb->centroid = eb->position;

    if (*version > -6) {
	int filter_count;
	if (!xdr_int(xdr, &filter_count)) return(FALSE);
	for(i = 0; i< MAX_FILTER_COUNT; i++) {
	    if (! xdr_int(xdr, &(eb->filter_id[i]))) return(XDR_ERROR);
	}
	if (filter_count == 1) {
	    if (eb->filter_id[0] == CUSTOM_FILTER_ID) {
		eb->filter_type[0] = COMPENSATOR_FILTER;
		eb->filter_type[1] = NO_FILTER;
		eb->filter_id[1] = NO_FILTER;
	    }
	    else if (eb->filter_id[0] == ANASTRUCT_FILTER_ID) {
		eb->filter_type[0] = NO_FILTER;
		eb->filter_type[1] = BOLUS_FILTER;
	    }
	    else if (eb->filter_id[1] >= 15 &&
		     eb->filter_id[1] <= 60) {
		eb->filter_type[0] = DYNAMIC_FILTER;
		eb->vw_angle = eb->filter_id[1];
		eb->filter_type[1] = NO_FILTER;
		eb->filter_id[1] = NO_FILTER;
	    }
	    else {
		eb->filter_type[0] = REAL_FILTER;
		eb->filter_type[1] = NO_FILTER;
		eb->filter_id[1] = NO_FILTER;
	    }
	}
	else {
	    if (eb->filter_id[0] == CUSTOM_FILTER_ID) {
		eb->filter_type[0] = CUSTOM_FILTER_ID2;
		eb->filter_id[0] = eb->filter_id[1];
		eb->filter_id[1] = NO_FILTER;
	    }
	    else {
		eb->filter_type[0] = NO_FILTER;
		eb->filter_id[0] = NO_FILTER;
		eb->filter_type[1] = NO_FILTER;
		eb->filter_id[1] = NO_FILTER;
	    }
	}
    }
    else {
	for(i = 0; i < MAX_FILTER_COUNT; i++) {
	    if (! xdr_int(xdr, &(eb->filter_type[i]))) return(XDR_ERROR);
	    if (! xdr_int(xdr, &(eb->filter_id[i]))) return(XDR_ERROR);
	}
	if (! xdr_int(xdr, &(eb->vw_angle))) return(XDR_ERROR);
    }
    if (!
	(xdr_int(xdr, &eb->tray_id) &&
	 xdr_float(xdr, &eb->start_angle) &&
	 xdr_float(xdr, &eb->stop_angle) &&
	 xdr_float(xdr, &eb->angle_inc)))
	return(FALSE);

    return(TRUE);
}

int
xdr_EXT_OLD_BEAM(XDR *xdr, EXT_BEAM *eb)
{
    char		*cp;
    unsigned int	 lp;
    int			i, j;
    int			version;
    int			filter_count = 0;

    cp = &(eb->name[0]);

    /*
    if (xdr->x_op == XDR_ENCODE &&
	eb->x_jaw_count == 1 && eb->y_jaw_count == 1)
	version = eb->unit_id;
    else if (xdr->x_op == XDR_ENCODE &&
	     eb->segment_count == 1)
	version = -1;
    else version = -6;
    */
    version = -2;
    bool_t serial_result;
#if defined(__APPLE__) && defined(__LP64__)
    {
    unsigned int serial;
    serial_result = xdr_u_long(xdr, &serial);
    eb->serial_number = serial;
    }
#else
    serial_result = xdr_u_long(xdr,&eb->serial_number);
#endif
    if (!(serial_result &&
	  xdr_string(xdr, &cp, NAME_LENGTH) &&
	  xdr_int(xdr, &version)))
	return(FALSE);
    if (version <= -2) {
	/* MLC with multiple segments has -2 and then unit_id */
	if (!xdr_int(xdr, &eb->unit_id))
	    return(FALSE);
	if (!(xdr_int(xdr, &eb->x_jaw_count) &&
	      xdr_int(xdr, &eb->y_jaw_count)))
	    return(FALSE);
	if (!(xdr_int(xdr, &eb->segment_count)))
	    return(FALSE);
    }
    else if (version == -1) {
	/* MLC has -1 and then unit_id */
	if (!xdr_int(xdr, &eb->unit_id))
	    return(FALSE);
	if (!(xdr_int(xdr, &eb->x_jaw_count) &&
	      xdr_int(xdr, &eb->y_jaw_count)))
	    return(FALSE);
	eb->segment_count = 1;
    }
    else {
	/* Reading old style with X and Y jaws */
	eb->unit_id = version;
	eb->x_jaw_count = 1;
	eb->y_jaw_count = 1;
	eb->segment_count = 1;
    }

    if (xdr->x_op == XDR_DECODE) {
	eb->segment_weight = (float *)
	    malloc(eb->segment_count*sizeof(float));
	eb->jaw = (JAW_POSITION **)
	    malloc(eb->segment_count*sizeof(JAW_POSITION *));
    }

    lp = eb->x_jaw_count + eb->y_jaw_count;

    for (j = 0; j < eb->segment_count; j++) {
      if (xdr->x_op == XDR_DECODE)
	eb->jaw[j] = (JAW_POSITION *)
	    malloc(MAX_JAW_COUNT*sizeof(JAW_POSITION));
      for(i = 0; i < static_cast<int>(lp); i++) {
	JAW_POSITION *jaw = eb->jaw[j] + i;
	if (version <= -2) {
	    if (!(xdr_float(xdr, &(eb->segment_weight[j]))))
		return(FALSE);
	}
	else eb->segment_weight[j] = 1.0;
	if (version <= -1) {
	    if (!(xdr_int(xdr, &(jaw->independent)) &&
		  xdr_float(xdr, &(jaw->pos_1)) &&
		  xdr_float(xdr, &(jaw->pos_2)) &&
		  xdr_float(xdr, &(jaw->min_w)) &&
		  xdr_float(xdr, &(jaw->max_w))))
		return(FALSE);
	}
	else {
	    if (!(xdr_int(xdr, &(jaw->independent)) &&
		  xdr_float(xdr, &(jaw->pos_1)) &&
		  xdr_float(xdr, &(jaw->pos_2))))
		return(FALSE);
	    jaw->min_w = -20.0;
	    jaw->max_w = 20.0;
	}
      }
    }

    if (!xdr_POINT(xdr, &eb->position)) return(FALSE);

    if (!(xdr_float(xdr, &eb->gantry_angle) &&
	  xdr_float(xdr, &eb->coll_angle) &&
	  xdr_float(xdr, &eb->table_angle) &&
	  xdr_float(xdr, &eb->SSD)))
	return(FALSE);

    for(i=0; i<4; i++)
	for(j=0; j<4; j++)
	    if (! xdr_float(xdr, &(eb->T_pat_to_beam[i][j])))
		return(FALSE);

    for(i=0; i<4; i++)
	for(j=0; j<4; j++)
	    if (! xdr_float(xdr, &(eb->T_beam_to_pat[i][j])))
		return(FALSE);

    if (xdr->x_op == XDR_ENCODE) {
	if (eb->filter_type[0] != NO_FILTER) filter_count = 1;
	else filter_count = 0;
    }

    if (!
	(xdr_int(xdr, &eb->placement) &&
	 xdr_int(xdr, &eb->extent) &&
	 xdr_int(xdr, &eb->corrections) &&
	 xdr_int(xdr, &eb->inh_type) &&
	 xdr_int(xdr, &filter_count)))
	return(FALSE);

    for(i=0; i<MAX_FILTER_COUNT; i++)
	if (! xdr_int(xdr, &(eb->filter_id[i])))
	    return(XDR_ERROR);
    if (xdr->x_op == XDR_DECODE) {
	if (filter_count == 0) eb->filter_type[0] = NO_FILTER;
    }

    if (!
	(xdr_int(xdr, &eb->tray_id) &&
	 xdr_float(xdr, &eb->start_angle) &&
	 xdr_float(xdr, &eb->stop_angle) &&
	 xdr_float(xdr, &eb->angle_inc)))
	return(FALSE);

    return(TRUE);
}

/*---------- IMPLANT_DESC ----------*/

/*FUNCTION: xdr_IMPLANT_DESC *************************************************

PURPOSE
XDR primitive for IMPLANT_DESC

***************************************************************************/
int
xdr_IMPLANT_DESC(XDR *xdr, IMPLANT_DESC *i)
{
    if (xdr_int(xdr, &i->source_count) &&
	xdr_int(xdr, &i->seed_count))
	return(TRUE);

    return(FALSE);
}

/*FUNCTION: xdr_SOURCE *************************************************

PURPOSE
XDR primitive for SOURCE

***************************************************************************/
int
xdr_SOURCE(XDR *xdr, SOURCE *s)
{
    unsigned int lp = 2;
    char *cp;

    cp = (char *) &(s->p[0]);
    if 	(xdr_int(xdr, &s->type) &&
	 xdr_array(xdr, &cp, &lp, 2, sizeof(PNT3D), (xdrproc_t)xdr_POINT))
	return(TRUE);

    return(FALSE);
}

/*FUNCTION: xdr_SEED *************************************************

PURPOSE
XDR primitive for SEED

***************************************************************************/
int
xdr_SEED(XDR *xdr, SEED *s)
{
    if (xdr_int(xdr, &(s->type)) &&
	xdr_POINT(xdr, &(s->p)))
	return(TRUE);

    return(FALSE);
}



/*---------- TWOD_GRID ----------*/

/*FUNCTION: xdr_TWOD_GRID *************************************************

PURPOSE
XDR primitive for TWOD_GRID

***************************************************************************/
int
xdr_TWOD_GRID(XDR *xdr, TWOD_GRID *g)
{
    int i, j;

    if (!
	(xdr_float(xdr, &g->x_start) &&
	 xdr_float(xdr, &g->y_start) &&
	 xdr_float(xdr, &g->x_inc) &&
	 xdr_float(xdr, &g->y_inc) &&
	 xdr_int(xdr, &g->x_count) &&
	 xdr_int(xdr, &g->y_count)))
	return(FALSE);

    for(i=0; i<4; i++)
	for(j=0; j<4; j++)
	    if (! xdr_float(xdr, &(g->grid_to_pat_T[i][j])))
		return(FALSE);

    for(i=0; i<4; i++)
	for(j=0; j<4; j++)
	    if (! xdr_float(xdr, &(g->pat_to_grid_T[i][j])))
		return(FALSE);
    
    if (!
	(xdr_float(xdr, &g->max) &&
	 xdr_float(xdr, &g->min)))
	return(FALSE);

    return(TRUE);
}


/*---------- THREED_GRID ----------*/

/*FUNCTION: xdr_GRID *************************************************

PURPOSE
XDR primitive for GRID

***************************************************************************/
int
xdr_GRID(XDR *xdr, GRID *g)
{
    int i, j;

    if (!
	(xdr_POINT(xdr, &g->start) &&
	 xdr_POINT(xdr, &g->inc) &&
	 xdr_int(xdr, &g->x_count) &&
	 xdr_int(xdr, &g->y_count) &&
	 xdr_int(xdr, &g->z_count) ))
	return(FALSE);

    for(i=0; i<4; i++)
	for(j=0; j<4; j++)
	    if (! xdr_float(xdr, &(g->grid_to_pat_T[i][j])))
		return(FALSE);

    for(i=0; i<4; i++)
	for(j=0; j<4; j++)
	    if (! xdr_float(xdr, &(g->pat_to_grid_T[i][j])))
		return(FALSE);
    
    if (!
	(xdr_float(xdr, &g->max) &&
	 xdr_float(xdr, &g->min)))
	return(FALSE);

    return(TRUE);
}

/*---------- WEIGHT ----------*/

/*FUNCTION: xdr_WEIGHTS *************************************************

PURPOSE
XDR primitive for WEIGHTS

***************************************************************************/
int
xdr_WEIGHTS(XDR *xdr, WEIGHTS *w)
{
    if (! xdr_int(xdr, &w->count)) return(FALSE);
    return(TRUE);
}

/*FUNCTION: xdr_WEIGHT *************************************************

PURPOSE
XDR primitive for WEIGHT

***************************************************************************/
int
xdr_WEIGHT(XDR *xdr, WEIGHT *w)
{
  // unused var // int i;
    char *cp;
    unsigned int lp;

    lp = 5;
    cp = (char *)&(w->f.wt[0]);

    if (xdr_int(xdr, &w->factors) &&
	xdr_array(xdr, &cp, &lp, 5, sizeof(float), (xdrproc_t)xdr_float))
	return(TRUE);
    else
	return(FALSE);

}

/*---------- CALC_POINTS ----------*/

/*FUNCTION: xdr_CALC_POINT *************************************************

PURPOSE
XDR primitive for CALC_POINT

***************************************************************************/
int
xdr_CALC_POINT(XDR *xdr, CALC_POINT *c)
{
    char *cp;

    cp = &(c->label[0]);
    if (xdr_string(xdr, &cp, 100) &&
	xdr_POINT(xdr, &c->p))
	return(TRUE);

    return(FALSE);
}

/*---------- PLAN ----------*/

/*FUNCTION: xdr_PLAN *************************************************

PURPOSE
XDR primitive for PLAN

***************************************************************************/
int
xdr_PLAN(XDR *xdrs, PLAN *p)
{   int i, j;
    int	temp;

    for (i=0; i<4; i++)
	for (j=0; j<4; j++)
	    if (! xdr_float(xdrs, &(p->pat_to_plan_T[i][j])))
		return(FALSE);

    /* old style has isodose_count only */
    /* new style has -1, isodose_count, ref_dose */
    temp = -1;
    if (! xdr_int(xdrs, &temp)) return(FALSE);

    if (temp == -1) { /* new style */
	if (! xdr_int(xdrs, &p->isodose_count)) return(FALSE);
	if (! xdr_float(xdrs, &p->ref_dose)) return(FALSE);
    }
    else { /* old style, force ref_dose to 100.0 */
	p->isodose_count = temp;
	p->ref_dose = 100.0;
    }

    return(TRUE);
}

/* ---------- PLAN_IM_HEADER ---------- */

/*FUNCTION: xdr_PLAN_IM_HEADER *************************************************

PURPOSE
XDR primitive for PLAN_IM_HEADER

***************************************************************************/
int
xdr_PLAN_IM_HEADER(XDR *xdr, plan_im_header *p, int write_flag)
{
    if (write_flag) return(xdr_write_PLAN_IM_HEADER(xdr, p));
    else return(xdr_read_PLAN_IM_HEADER(xdr, p));
}

int
xdr_write_PLAN_IM_HEADER(XDR *xdr, plan_im_header *p)
{   int		opcode;
    int		oplen;
    char	*cp;
    // unused var //char	temp_unit[20];
    
    if (xdr_read_write_PLAN_IM_HEADER(xdr, p, 1) == FALSE) return(FALSE);

    opcode = PLAN_IM_FIRST_OPCODE;
    if (! xdr_int(xdr, &opcode)) return(FALSE);
    oplen = 0;
    if (! xdr_int(xdr, &oplen)) return(FALSE);
    cp = &(p->unit_number[0]);
    if (strlen(cp) > 8) {
	opcode = PLAN_IM_UNIT_NUMBER;
	if (! xdr_int(xdr, &opcode)) return(FALSE);
	oplen = (4 + ((strlen(cp) + 3)>>2))<<2;
	if (! xdr_int(xdr, &oplen)) return(FALSE);
	if (! xdr_string(xdr, &cp, 20)) return(FALSE);
    }

    if (p->y_dim != p->x_dim) {
	opcode = PLAN_IM_Y_DIM;
	if (! xdr_int(xdr, &opcode)) return(FALSE);
	oplen = 4;
	if (! xdr_int(xdr, &oplen)) return(FALSE);
	if (! xdr_int(xdr, &p->y_dim)) return(FALSE);
    }

    if (p->y_size != p->x_size) {
	opcode = PLAN_IM_Y_SIZE;
	if (! xdr_int(xdr, &opcode)) return(FALSE);
	oplen = 4;
	if (! xdr_int(xdr, &oplen)) return(FALSE);
	if (! xdr_float(xdr, &p->y_size)) return(FALSE);
    }

    opcode = PLAN_IM_END;
    if (! xdr_int(xdr, &opcode)) return(FALSE);
    oplen = 0;
    if (! xdr_int(xdr, &oplen)) return(FALSE);

    return(TRUE);
}

int
xdr_read_PLAN_IM_HEADER(XDR *xdr, plan_im_header *p)
{   int		i;
    int		opcode;
    int		oplen;
    int		min_pos;
    char	*cp;

    if (xdr_read_write_PLAN_IM_HEADER(xdr, p, 0) == FALSE) return(FALSE);
    p->x_dim = p->y_dim = p->resolution;
    p->x_size = p->y_size = p->pixel_size;

    /* Check for the temporary way of handling large unit numbers */
    cp = &(p->unit_number[0]);
    if (strncmp(cp, "PLUNC_1", 7) == 0) {
	if (! xdr_string(xdr, &cp, 20)) return(FALSE);
    }

    /* Now process any optional opcodes */
    /* Find the first offset pointer */
    min_pos = p->per_scan[0].offset_ptrs;
    for (i = 1; i < p->slice_count; i++) {
	if (min_pos > p->per_scan[i].offset_ptrs)
	    min_pos = p->per_scan[i].offset_ptrs;
    }
    if (xdr_pos(xdr) >= min_pos) return(TRUE);

    /* Check the first opcode */
    if (!xdr_int(xdr, &opcode)) return(FALSE);
    if (opcode == 0) return(TRUE);
    if (opcode != PLAN_IM_FIRST_OPCODE) return(TRUE);
    if (!xdr_int(xdr, &oplen)) return(FALSE);
    if (oplen != 0) return(FALSE);

    /* Check remaining opcodes */
    while (xdr_pos(xdr) < min_pos) {
	if (!xdr_int(xdr, &opcode)) return(FALSE);
	if (!xdr_int(xdr, &oplen)) return(FALSE);
	switch (opcode) {
	    case PLAN_IM_UNIT_NUMBER :
		cp = &(p->unit_number[0]);
		if (! xdr_string(xdr, &cp, 20)) return(FALSE);
		break;
	    case PLAN_IM_Y_DIM :
		if (oplen != 4) return(FALSE);
		if (! xdr_int(xdr, &p->y_dim)) return(FALSE);
		break;
	    case PLAN_IM_Y_SIZE :
		if (oplen != 4) return(FALSE);
		if (! xdr_float(xdr, &p->y_size)) return(FALSE);
		break;
	    case PLAN_IM_END :
		if (oplen != 0) return(FALSE);
		return(TRUE);
	    default :
		for (i = 0; i < oplen; i += 4) {
		    if (!xdr_int(xdr, &opcode)) return(FALSE);
		}
		break;
	}
    }

    return(TRUE);
}

int
xdr_read_write_PLAN_IM_HEADER(XDR *xdr, plan_im_header *p, int write_flag)
{   int		i, j;
// unused var//int		opcode;
// unused var//int		min_pos;
    char	*cp;
    char	temp_unit[20];
    unsigned	lp;

    if (write_flag) {
	strncpy(temp_unit, p->unit_number, 8);
	temp_unit[8] = 0;
	cp = &(temp_unit[0]);
    }
    else cp = &(p->unit_number[0]);
    if (! xdr_string(xdr, &cp, 9)) return(FALSE);

    cp = &(p->patient_name[0]);
    if (! xdr_string(xdr, &cp, 100)) return(FALSE);
    
    if (! xdr_DATE(xdr, &p->date)) return(FALSE);

    cp = &(p->comment[0]);
    if (! (
	   xdr_string(xdr, &cp, 100) &&
	   xdr_enum(xdr, (int *)&p->machine_id) &&
	   xdr_enum(xdr, (int *)&p->patient_position) &&
	   xdr_enum(xdr, (int *)&p->whats_first) &&
	   xdr_enum(xdr, (int *)&p->pixel_type) &&
	   xdr_int(xdr, (int *)&p->slice_count) &&
	   xdr_int(xdr, (int *)&p->resolution) &&
	   xdr_float(xdr, &p->pixel_size) &&
	   xdr_int(xdr, &p->table_height) &&
	   xdr_int(xdr, &p->min) &&
	   xdr_int(xdr, &p->max)))
	return(FALSE);

    for (i=0; i<4; i++)
	for (j=0; j<4; j++)
	    if (! xdr_float(xdr, &(p->pixel_to_patient_TM[i][j])))
		return(FALSE);

    cp = (char *) &(p->per_scan[0]);
    if (write_flag) {
        lp = 500; //ABSURDLY_LARGE_MAX_SCANS;
			// must be 500 to be readable by standard PLUNC
    } else {
        lp = p->slice_count;
    }
		
	// original version as seen in standard PLUNC
	// fails because ABSURDLY_LARGE_MAX_SCANS is
	// 1000 in CompOnc and so array size does
	// not fit (-gst 2004-feb-09)
/*
    lp = ABSURDLY_LARGE_MAX_SCANS;
    if (! xdr_array(xdr, &cp, &lp, ABSURDLY_LARGE_MAX_SCANS,
		    sizeof(per_scan_info), (xdrproc_t)xdr_PER_SCAN_INFO))
	return(FALSE);
*/

	// ignore xdr_array's count for downward compatibility;
	// not strictly needed as this is the more advanced
	// version
    if (write_flag) {
		xdr_int(xdr, &(p->slice_count));
	} else {
		int junk = p->slice_count;
		xdr_int(xdr, &junk);
	}

    for (int scan = 0; scan < p->slice_count; scan++)
    {
       if (! xdr_PER_SCAN_INFO(xdr, &p->per_scan[scan])) {
         printf ("xdr_PER_SCAN_INFO failed in scan %d\n", scan);
         return(FALSE);
       }
    }
    return(TRUE);
}

/* ---------- PER_SCAN_INFO ---------- */

/*FUNCTION: xdr_PER_SCAN_INFO *************************************************

PURPOSE
XDR primitive for PER_SCAN_INFO

***************************************************************************/
int
xdr_PER_SCAN_INFO(XDR *xdr, per_scan_info *s)
{
    if (xdr_float(xdr, &s->z_position) &&
	xdr_int(xdr, &s->offset_ptrs) &&
	xdr_float(xdr, &s->gantry_angle) &&
	xdr_float(xdr, &s->table_angle) &&
	xdr_int(xdr, &s->scan_number))
	return(TRUE);
    else
	return(FALSE);
}

/* ---------- DATE ---------- */

/*FUNCTION: xdr_DATE *************************************************

PURPOSE
XDR primitive for DATE

***************************************************************************/
int
xdr_DATE(XDR *xdr, PDATE  *date)
{
    if ((xdr_int(xdr, &date->day)) &&
	(xdr_int(xdr, &date->month)) &&
	(xdr_int(xdr, &date->year)) &&
	(xdr_int(xdr, &date->dow)))
	return(TRUE);
    else
	return(FALSE);
}

/* ---------- SCAN  ---------- */

/*FUNCTION: xdr_SCAN *************************************************

PURPOSE
XDR primitive for SCAN

***************************************************************************/
int
xdr_SCAN( XDR *xdr, short *scan, int resolution)
{
#ifdef USE_XDR_ARRAY
    unsigned int lp;
    boolt_t xdr_short();

    lp = resolution * resolution ;

    if (! xdr_array(xdr, &scan, &lp, lp, sizeof(short),
		   (xdrproc_t)xdr_short))
	return(FALSE);
    
    return(TRUE);
#else
    fprintf(stderr, "\nxdr_SCAN in plan_xdr_defs.c not compiled properly\n");

    return(FALSE);
#endif
}

/* ---------- BRACHY_OBJECT ---------- */

/*FUNCTION: xdr_ll_brachy_object *************************************************

PURPOSE
XDR primitive for ll_brachy_object

***************************************************************************/
int
xdr_ll_brachy_object(XDR_fdes *xdr_fdes, BRACHY_OBJECT *o1, int read_flag)
{
    int j;
    char *cp;
    int status = XDR_NO_ERROR;
    int size;

    cp = &(o1->label[0]);
    if (!
	(xdr_string(xdr_fdes->xdrs, &cp, 100) &&
	 xdr_int(xdr_fdes->xdrs, &o1->seed_count) &&
	 xdr_int(xdr_fdes->xdrs, &o1->source_count)))
    {
	XDR_ReportError("Cannot process brachy_object header");
	status = XDR_ERROR;
	goto cleanup;
    }


    if (read_flag && o1->seed_count)
    {
	size = o1->seed_count * sizeof(unsigned int);
	o1->seed_list = (Boolean *) malloc(size);
	if (o1->seed_list == NULL)
	{
	    XDR_ReportError("Cannot malloc brachy object seed list");
	    status = XDR_ERROR;
	    goto cleanup;
	    }
    }

    for(j=0; j<o1->seed_count; j++)
	if (! xdr_int(xdr_fdes->xdrs, &(o1->seed_list[j])))
	{
	    XDR_ReportError("Cannot process brachy_object seed");
	    status = XDR_ERROR;
	    goto cleanup;
	}

    if (read_flag && o1->source_count)
    {
	size = o1->source_count * sizeof(Boolean);
	o1->source_list = (Boolean *) malloc(size);
	if (o1->source_list == NULL)
	{
	    XDR_ReportError("Cannot malloc brachy object source list");
	    status = XDR_ERROR;
	    goto cleanup;
	}
    }

    for(j=0; j<o1->source_count; j++) {
	if (! xdr_int(xdr_fdes->xdrs, &(o1->source_list[j]))) {
	    XDR_ReportError("Cannot process brachy_object seed");
	    status = XDR_ERROR;
	    goto cleanup;
	}
    }

 cleanup:
    return(status);
}

/* ---------- SEED_SPEC ---------- */

/*FUNCTION: xdr_SEED_SPEC *************************************************

PURPOSE
XDR primitive for SEED_SPEC

***************************************************************************/
int
xdr_SEED_SPEC( XDR *xdr, SEED_SPEC *s)
{
    int		i;
    char	*cp;

    cp = &(s->isotope[0]);
    if (!
	(xdr_string(xdr, &cp, 50) &&
	 xdr_float(xdr, &s->gamma) &&
	 xdr_int(xdr, &s->gammaUnits) &&
	 xdr_float(xdr, &s->R_to_r) &&
	 xdr_float(xdr, &s->half_life) &&
	 xdr_int(xdr, &s->TA_count)))
	return(FALSE);
	 
    for(i=0; i<20; i++)
	if (! xdr_float(xdr, &(s->tissue_attenuation[i])))
	    return(FALSE);

    for(i=0; i<20; i++)
	if (! xdr_float(xdr, &(s->TA_distance[i])))
	    return(FALSE);

    if (!
	(xdr_float(xdr, &s->mu) &&
	 xdr_int(xdr, &s->last_entry)))
	return(FALSE);

    for(i=0; i<SEED_RADII; i++)
	if (! xdr_float(xdr, &(s->seed_dose_table[i])))
	    return(FALSE);

    return(TRUE);
}

