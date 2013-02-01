
#ifndef _PLUNC_HAVE_LIBPLANIO_H
#define _PLUNC_HAVE_LIBPLANIO_H

#include "gen.h"

#ifdef WIN32
#include "xdr.h"
#include "getopt.h"
struct passwd {
    char *pw_dir;
};
int geteuid();
passwd *getpwuid(int);

#else

#include <pwd.h>

#endif

/*
#ifdef __cplusplus
extern "C" {
#endif
*/


#ifdef _PLUNC_HAVE_EXTBM_H
int get_filter(char *phys_dat_dir, int filter_id, FILTER *filter);
#endif

char *get_phys_dat_dir();

char *get_plunc_root_dir();

#ifdef _PLUNC_HAVE_PLAN_IM_H
int
get_slice(int fd, plan_im_header *im_header,
	  int start_slice, int end_slice, PIXELTYPE *buffer);

int
put_slice(int fd, plan_im_header *im_header,
	   int start_slice, int end_slice, PIXELTYPE *buffer);
#endif

#ifdef _PLUNC_HAVE_EXTBM_H
int
get_time_calc(char *phys_dat_dir, int unit_id,
	      TIME_CALC_PARAMETERS *time_calc);
int
get_tray(
    char        *phys_dat_dir,
    int         tray_id,
    TRAY        *tray);

int
get_unit(
    char        *phys_dat_dir,
    int         unit_id,
    UNIT        *unit);
#endif

#ifdef _PLUNC_HAVE_PLAN_XDR_H
int
xdr_ANASTRUCT(
    XDR         *xdr,
    ANASTRUCT   *an);

int
xdr_CONTOUR(
    XDR *xdr,
    CONTOUR *contour);

int
xdr_CONTOUR_X_Y(
    XDR *xdr,
    float *xy,
    int count);

int
xdr_POINT(
    XDR *xdr,
    PNT3D *point);

int
xdr_Boolean(
    XDR *xdr,
    Boolean *b);

#ifdef _PLUNC_HAVE_EXTBM_H
int
xdr_EXT_BEAM(
    XDR         *xdr,
    EXT_BEAM    *eb,
    int		*version);

int
xdr_EXT_OLD_BEAM(
    XDR         *xdr,
    EXT_BEAM    *eb);
#endif

#ifdef _PLUNC_HAVE_BRACHY_H
int
xdr_IMPLANT_DESC(
    XDR *xdr,
    IMPLANT_DESC *i);

int
xdr_SOURCE(
    XDR *xdr,
    SOURCE *s);

int
xdr_SEED(
    XDR *xdr,
    SEED *s);
#endif

int
xdr_TWOD_GRID(
    XDR *xdr,
    TWOD_GRID *g);

int
xdr_GRID(
    XDR *xdr,
    GRID *g);

int
xdr_WEIGHTS(
    XDR *xdr,
    WEIGHTS *w);

int
xdr_WEIGHT(
    XDR *xdr,
    WEIGHT *w);

int
xdr_CALC_POINT(
    XDR *xdr,
    CALC_POINT *c);

int
xdr_PLAN(
    XDR *xdrs,
    PLAN *p);

#ifdef _PLUNC_HAVE_PLAN_IM_H
int
xdr_PLAN_IM_HEADER(
    XDR                 *xdr,
    plan_im_header      *p,
    int                 write_flag);

int
xdr_write_PLAN_IM_HEADER(
    XDR                 *xdr,
    plan_im_header      *p);

int
xdr_read_PLAN_IM_HEADER(
    XDR                 *xdr,
    plan_im_header      *p);

int
xdr_read_write_PLAN_IM_HEADER(
    XDR                 *xdr,
    plan_im_header      *p,
    int			write_flag);

int
xdr_PER_SCAN_INFO(
    XDR *xdr,
    per_scan_info *s);
#endif

int
xdr_DATE(
    XDR *xdr,
    PDATE  *date);

int
xdr_SCAN( XDR *xdr, short *scan, int resolution);

#ifdef _PLUNC_HAVE_BRACHY_H
int
xdr_ll_brachy_object(XDR_fdes *xdr_fdes, BRACHY_OBJECT *o1, int read_flag);

int
xdr_SEED_SPEC( XDR *xdr, SEED_SPEC *s);
#endif

#endif

int
read_2d_grid(int fdes, TWOD_GRID *g, int desc_only);

int
write_2d_grid(int fdes, TWOD_GRID *g, int desc_only);

int
read_anastruct(int fdes, ANASTRUCT *anastruct_desc);

int
write_anastruct(int fdes, ANASTRUCT *anastruct_desc);

#ifdef _PLUNC_HAVE_EXTBM_H
int
read_beam(int fdes, EXT_BEAM *eb);

int
write_beam(int fdes, EXT_BEAM *eb);

int
write_old_beam(int fdes, EXT_BEAM *eb);
#endif

int read_byte(int fdes, char *ptr);
int write_byte(int fdes, char *ptr);
int read_bytes(int fdes, char *ptr, int num);
int write_bytes(int fdes, char *ptr, int num);
int read_write_byte(int fdes, char *ptr, char mode);
int read_int(int fdes, int * ptr);
int write_int(int fdes, int *ptr);
int read_ints(int fdes, int *ptr, int num);
int write_ints(int fdes, int *ptr, int num);
int read_write_ints(int fdes, int *ptr, int num, char mode);
int read_short(int fdes, short *ptr);
int write_short(int fdes, short *ptr);
int read_shorts(int fdes, short *ptr, int num);
int write_shorts(int fdes, short *ptr, int num);
int read_write_short(int fdes, short *ptr, char mode);
int read_float(int fdes, float *ptr);
int write_float(int fdes, float *ptr);
int read_floats(int fdes, float *ptr, int num);
int write_floats(int fdes, float *ptr, int num);
int read_write_floats(int fdes, float *ptr, int num, char mode);
int read_byte_array(int fdes, char *ptr, int num);
int write_byte_array(int fdes, char *ptr, int num);

int read_contour(int fdes, CONTOUR *c);
int write_contour(int fdes, CONTOUR *c);

#ifdef _PLUNC_HAVE_EXTBM_H
int read_filter(int fdes, FILTER *filter);
int write_filter(int fdes, FILTER *filter);
#endif

int read_grid(int fdes, GRID *g, int desc_only);
int write_grid(int fdes, GRID *g, int desc_only);

#ifdef _PLUNC_HAVE_PLAN_IM_H
int read_image_header(int fdes, plan_im_header *header);
int write_image_header(int fdes, plan_im_header *header);
#endif

#ifdef _PLUNC_HAVE_BRACHY_H
int read_implant_desc(int fdes, IMPLANT_DESC *implant_desc);
int read_object(int fdes, BRACHY_OBJECT *object_desc);
int read_objects(int fdes, BRACHY_OBJECTS *objects_desc);
#endif

#ifdef _PLUNC_HAVE_EXTBM_H
int read_filters(int *count, FILTER **f);
int read_trays(int *count, TRAY **t);
int read_units(int *count, UNIT **u);
#endif

int read_map(char *name, THREED_MAP *map, int slice, int num_slices);
int write_map(char *name, THREED_MAP *map, int slice, int num_slices);

int read_write_norm(int fdes, NORM *norm, char mode);
int write_norm(int fdes, NORM *norm);

int read_plan(int fdes, PLAN *p);
int write_plan(int fdes, PLAN *p);
int read_plans(int fdes, PLAN *plan_desc);
int write_plans(int fdes, PLAN *plan_desc, int num);

int read_points(int fdes, CALC_POINTS *p, int desc_only);
int write_points(int fdes, CALC_POINTS *p, int desc_only);

#ifdef _PLUNC_HAVE_EXTBM_H
int read_sar(int fdes, SAR_TABLE *sar);
int write_sar(int fdes, SAR_TABLE *sar);
#endif

int read_scan(int fdes, short *scan, int offset, int res);
int read_scan_xy(int fdes, short *scan, int offset, int xdim, int ydim);
int write_scan(int fdes, short *scan, int offset, int res);
int write_scan_xy(int fdes, short *scan, int offset, int xdim, int ydim);
void my_swab(char *src, char *dest, int num);

#ifdef _PLUNC_HAVE_EXTBM_H
int read_time_calc(int fdes, TIME_CALC_PARAMETERS *time_calc);
int write_time_calc(int fdes, TIME_CALC_PARAMETERS *time_calc);
int read_tray(int fdes, TRAY *tray);
int write_tray(int fdes, TRAY *tray);
int read_unit(int fdes, UNIT *unit);
int write_unit(int fdes, UNIT *unit);
#endif

int read_norm(int fdes, NORM *norm);
int write_norm(int fdes, NORM *norm);

int read_weights(int fdes, WEIGHTS *w);
int write_weights(int fdes, WEIGHTS *w);

void *Realloc(void *ptr, int size);

int sock_read(int fdes, char *buff, unsigned int buff_size);
int sock_write(int fdes, char *buff, unsigned int buff_size);

int runlen_encode(char *filename, RGB_PIXEL *buf, int xdim, int ydim);

#ifdef _PLUNC_HAVE_BRACHY_H
int write_implant_desc(int fdes, IMPLANT_DESC *implant_desc);
int write_object(int fdes, BRACHY_OBJECT *object_desc);
int write_objects(int fdes, BRACHY_OBJECTS *objects_desc);
#endif

int fd_poll(int fd, int msec);

#ifdef _PLUNC_HAVE_PLAN_XDR_H
bool_t xdrfd_get_bytes(XDR *xdrs, caddr_t dest, int count);
bool_t xdrfd_put_bytes(XDR *xdrs, caddr_t src, int count);
bool_t xdr_eof(int fdes);
void xdrfd_create(XDR *xdrs, int fd, enum xdr_op op);
int xdr_pos(XDR *xdrs);

bool_t xdrfd_flush_buff(XDR *xdrs);

XDR_fdes * xdr_ll_open_stream(int fdes, char mode);
void xdr_ll_close_stream(XDR_fdes *xdr_fdes);
void xdr_close(int fdes);
void XDR_ReportError(char *str);
void xdr_ll_flush(XDR_fdes *xdr_fdes);
#endif

int get_lantis_data(LANTIS_DATA *lantis);

void set_socket(int fd);
void clear_socket(int fd);
int is_socket(int fd);
int close_socket(int fd);

/*
#ifdef __cplusplus
}
#endif
*/

#endif

