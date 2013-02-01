
#ifdef KERNEL
#include "param.h"
#endif

#include <syslog.h>
#include <stdio.h>

#include <rpc/types.h>
#include <rpc/xdr.h>

#ifdef PLAN_MACHTEN_4_0_1
#define mc68000
#endif

/*
 * NB: Not portable.
 * This routine works on Suns (Sky / 68000's) and Vaxen.
 */
   
#ifdef vax

/* What IEEE single precision floating point looks like on a Vax */
struct  ieee_single {
	unsigned int	mantissa: 23;
	unsigned int	exp     : 8;
	unsigned int	sign    : 1;
};

/* Vax single precision floating point */
struct  vax_single {
	unsigned int	mantissa1 : 7;
	unsigned int	exp       : 8;
	unsigned int	sign      : 1;
	unsigned int	mantissa2 : 16;
		
};

#define VAX_SNG_BIAS    0x81
#define IEEE_SNG_BIAS   0x7f

static struct sgl_limits {
	struct vax_single s;
	struct ieee_single ieee;
} sgl_limits[2] = {
	{{ 0x7f, 0xff, 0x0, 0xffff },	/* Max Vax */
	{ 0x0, 0xff, 0x0 }},		/* Max IEEE */
	{{ 0x0, 0x0, 0x0, 0x0 },	/* Min Vax */
	{ 0x0, 0x0, 0x0 }}		/* Min IEEE */
};
#endif /* vax */

bool_t 
xdr_float(XDR *xdrs, float *fp)
{
#ifndef mc68000
	struct ieee_single is;
	struct vax_single vs, *vsp;
	struct sgl_limits *lim;
	int i;
#endif
	switch (xdrs->x_op) {

	case XDR_ENCODE:
#ifdef mc68000
		return (XDR_PUTLONG(xdrs, (long *)fp));
#else
		vs = *((struct vax_single *)fp);
		for (i = 0, lim = sgl_limits;
			i < sizeof(sgl_limits)/sizeof(struct sgl_limits);
			i++, lim++) {
			if ((vs.mantissa2 == lim->s.mantissa2) &&
				(vs.exp == lim->s.exp) &&
				(vs.mantissa1 == lim->s.mantissa1)) {
				is = lim->ieee;
				goto shipit;
			}
		}
		is.exp = vs.exp - VAX_SNG_BIAS + IEEE_SNG_BIAS;
		is.mantissa = (vs.mantissa1 << 16) | vs.mantissa2;
	shipit:
		is.sign = vs.sign;
		return (XDR_PUTLONG(xdrs, (long *)&is));
#endif

	case XDR_DECODE:
#ifdef mc68000
		return (XDR_GETLONG(xdrs, (long *)fp));
#else
		vsp = (struct vax_single *)fp;
		if (!XDR_GETLONG(xdrs, (long *)&is))
			return (FALSE);
		for (i = 0, lim = sgl_limits;
			i < sizeof(sgl_limits)/sizeof(struct sgl_limits);
			i++, lim++) {
			if ((is.exp == lim->ieee.exp) &&
				(is.mantissa == lim->ieee.mantissa)) {
				*vsp = lim->s;
				goto doneit;
			}
		}
		vsp->exp = is.exp - IEEE_SNG_BIAS + VAX_SNG_BIAS;
		vsp->mantissa2 = is.mantissa;
		vsp->mantissa1 = (is.mantissa >> 16);
	doneit:
		vsp->sign = is.sign;
		if ((vsp->exp == 0) && vsp->sign)
		{
		    /* for debugging only, lets record the event */
/*
		    syslog(LOG_ERR,
			   "xdr_float : %d %d %d %d\n",
			   vsp->sign, vsp->exp,
			   vsp->mantissa1, vsp->mantissa2);

		    fprintf(stderr,
			   "xdr_float : %1d %8d %7d %16d\n",
			   vsp->sign, vsp->exp,
			   vsp->mantissa1, vsp->mantissa2);
*/
		    vsp->sign = 0;
		}
/* JST - coerce sign bit to 0 if exponent is 0 - this is an illegal */
/* floating point representation on VAXen */
		return (TRUE);
#endif

	case XDR_FREE:
		return (TRUE);
	}
	return (FALSE);
}

/*
 * This routine works on Suns (Sky / 68000's) and Vaxen.
 */

#ifdef vax
/* What IEEE double precision floating point looks like on a Vax */
struct  ieee_double {
	unsigned int	mantissa1 : 20;
	unsigned int	exp	  : 11;
	unsigned int	sign	  : 1;
	unsigned int	mantissa2 : 32;
};
 
/* Vax double precision floating point */
struct  vax_double {
	unsigned int	mantissa1 : 7;
	unsigned int	exp       : 8;
	unsigned int	sign      : 1;
	unsigned int	mantissa2 : 16;
	unsigned int	mantissa3 : 16;
	unsigned int	mantissa4 : 16;
};

#define VAX_DBL_BIAS    0x81
#define IEEE_DBL_BIAS   0x3ff
#define MASK(nbits)     ((1 << nbits) - 1)

static struct dbl_limits {
	struct  vax_double d;
	struct  ieee_double ieee;
} dbl_limits[2] = {
	{{ 0x7f, 0xff, 0x0, 0xffff, 0xffff, 0xffff },	/* Max Vax */
	{ 0x0, 0x7ff, 0x0, 0x0 }},			/* Max IEEE */
	{{ 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},		/* Min Vax */
	{ 0x0, 0x0, 0x0, 0x0 }}				/* Min IEEE */
};

#endif /* vax */


bool_t           
xdr_double(XDR *xdrs, double *dp)
{
	register long *lp;
#ifndef mc68000
	struct  ieee_double id;
	struct  vax_double vd;
	register struct dbl_limits *lim;
	int i;
#endif

	switch (xdrs->x_op) {

	case XDR_ENCODE:
#ifdef mc68000
		lp = (long *)dp;
#else
		vd = *((struct  vax_double *)dp);
		for (i = 0, lim = dbl_limits;
			i < sizeof(dbl_limits)/sizeof(struct dbl_limits);
			i++, lim++) {
			if ((vd.mantissa4 == lim->d.mantissa4) &&
				(vd.mantissa3 == lim->d.mantissa3) &&
				(vd.mantissa2 == lim->d.mantissa2) &&
				(vd.mantissa1 == lim->d.mantissa1) &&
				(vd.exp == lim->d.exp)) {
				id = lim->ieee;
				goto shipit;
			}
		}
		id.exp = vd.exp - VAX_DBL_BIAS + IEEE_DBL_BIAS;
		id.mantissa1 = (vd.mantissa1 << 13) | (vd.mantissa2 >> 3);
		id.mantissa2 = ((vd.mantissa2 & MASK(3)) << 29) |
				(vd.mantissa3 << 13) |
				((vd.mantissa4 >> 3) & MASK(13));
	shipit:
		id.sign = vd.sign;
		lp = (long *)&id;
#endif
		return (XDR_PUTLONG(xdrs, lp++) && XDR_PUTLONG(xdrs, lp));

	case XDR_DECODE:
#ifdef mc68000
		lp = (long *)dp;
		return (XDR_GETLONG(xdrs, lp++) && XDR_GETLONG(xdrs, lp));
#else
		lp = (long *)&id;
		if (!XDR_GETLONG(xdrs, lp++) || !XDR_GETLONG(xdrs, lp))
			return (FALSE);
		for (i = 0, lim = dbl_limits;
			i < sizeof(dbl_limits)/sizeof(struct dbl_limits);
			i++, lim++) {
			if ((id.mantissa2 == lim->ieee.mantissa2) &&
				(id.mantissa1 == lim->ieee.mantissa1) &&
				(id.exp == lim->ieee.exp)) {
				vd = lim->d;
				goto doneit;
			}
		}
		vd.exp = id.exp - IEEE_DBL_BIAS + VAX_DBL_BIAS;
		vd.mantissa1 = (id.mantissa1 >> 13);
		vd.mantissa2 = ((id.mantissa1 & MASK(13)) << 3) |
				(id.mantissa2 >> 29);
		vd.mantissa3 = (id.mantissa2 >> 13);
		vd.mantissa4 = (id.mantissa2 << 3);
	doneit:
		vd.sign = id.sign;
		*dp = *((double *)&vd);
		return (TRUE);
#endif

	case XDR_FREE:
		return (TRUE);
	}
	return (FALSE);
}
