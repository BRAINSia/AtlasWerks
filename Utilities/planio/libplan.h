
#ifndef _PLUNC_HAVE_LIBPLAN_H
#define _PLUNC_HAVE_LIBPLAN_H

#include <stdio.h>

#define CCW_SENSE	1
#define CW_SENSE	-1

/*
#ifdef __cplusplus
extern "C" {
#endif
*/

#ifdef PLAN_WINNT
#define rint(x) (int)((x) + 0.5)
#endif

int
FindNtrsc(
    CONTOUR *Contour,
    float Point1[2], 
    float Point2[2],
    float Intersects[2][FI_LIMIT], 
    float Distances[FI_LIMIT]);

int
ana_ntrsc(
    ANASTRUCT   *obj,
    PNT3D       p1,
    PNT3D       p2,
    float       *alpha,
    int         debug);

float
compensator_ntrsc(PNT3D pt, ANASTRUCT *ana);

int anastruct_min_max(ANASTRUCT *ana);

int anastruct_normal(ANASTRUCT *ana, PNT3D pt, PNT3D *norm);

#ifdef _PLUNC_HAVE_EXTBM_H
void
beam_bolus(
    EXT_BEAM    *beam,
    ANASTRUCT   *ana,
    ANASTRUCT   *bolus,
    float       thickness,
    float       resolution,
    float       border);
#endif

int bin_search(float value, float *vector, int vec_size);

#ifdef _PLUNC_HAVE_EXTBM_H
int cmp_beams(EXT_BEAM *bm1, EXT_BEAM *bm2);
int collimate(EXT_BEAM *beam);
int collimator_enclosure(EXT_BEAM *beam);
int check_ML_collimator(EXT_BEAM *beam);
#endif

int contour_min_max(CONTOUR *c);

#ifdef _PLUNC_HAVE_PLAN_IM_H
void
con_image_bounding_box(
    plan_im_header *h,
    CONTOUR *con,
    int *min_i,
    int *max_i,
    int *min_j,
    int *max_j);

void
flag_above_threshold(
    plan_im_header      *header,
    PIXELTYPE           *image,
    int                 threshold);

void
flag_inside_ana(
    plan_im_header      *header,
    PIXELTYPE           *image,
    ANASTRUCT		*ana);

void
set_threshold_bit(
    plan_im_header      *header,
    PIXELTYPE           *image,
    int                 threshold,
    int                 bit);

int
fake_uniform_spacing(
    plan_im_header      *header,
    int                 **ptr,
    float               scale);
#endif

void
new_grid_slicer(
    GRID	*grid,
    float	pat_to_view_T[4][4],
    float	x_inc,
    float	y_inc,
    GRID	*grid2,
    int		alloc_flag);

int flesh_out_grid(GRID *old_grid, GRID *new_grid);
void get_date(PDATE *date);
char ** get_dirs(char *name, char *suffix);
char ** get_dirs2(char *name, char *suffix);
void free_dirs(char **dir_names);

#ifdef _PLUNC_HAVE_EXTBM_H
void
get_machine_senses(
    UNIT        *unit,
    float       *c_sense, 
    float       *c_off, 
    float       *g_sense, 
    float       *g_off, 
    float       *t_sense, 
    float       *t_off);
#endif

void get_serial_num(unsigned long *num);

void grid_min_max(GRID *g);
int in_ana(ANASTRUCT *ana, PNT3D point);

void
init_index_dose(
    GRID        *grid,
    DOSE_INDEX  **index_buff,
    float       val,
    float       weight);

void
set_index_dose(
    GRID        *grid,
    DOSE_INDEX  *index_buff,
    ANASTRUCT   *ana,
    float       ana_T[4][4],
    float       val,
    float       weight);

void
set_index_hist(
    GRID        *grid,
    DOSE_INDEX  *index_buff,
    ANASTRUCT   *ana,
    float       ana_T[4][4],
    int         num_hist,
    float       *hist_x,
    float       *hist_y,
    float       weight,
    int         flag);

void
set_index_hist2(
    DVH		*hist,
    USER_DVH	*target,
    DVH		*index_list,
    float	ref_dose,
    int		sensitive);

int
sample_index_dose(
    GRID        *grid,
    DOSE_INDEX  *index_buff,
    PNT3D       pt,
    float       *ret,
    float       *cnt);

void
merge_index_dose(
    GRID        *grid,
    DOSE_INDEX  *index_buff,
    float       *out);

int
sample_merged_dose(
    GRID        *grid,
    PNT3D       pt,
    float       *ret,
    float       *cnt);

int inout(CONTOUR *con, PNT3D point);

float
interp(
    int mode,
    float x1, 
    float x2, 
    float x, 
    float y1, 
    float y2, 
    float *fx);

void
least_squares_1D(
    float       *x,
    float       *y,
    int         dim,
    float       *output,
    int         degree);

#ifdef _PLUNC_HAVE_PLAN_IM_H
void
plan_im_sizer(
    plan_im_header	*im,
    LIB3D_TRANSFORM	pat_to_plan_T,
    LIB3D_TRANSFORM	inv_image_T,
    int			xres,
    int			yres,
    float		*xo,
    float		*yo,
    float		*xf,
    float		*yf);

void
plan_im_slicer(
    plan_im_header	*im,
    LIB3D_TRANSFORM	pat_to_plan_T,
    LIB3D_TRANSFORM	inv_image_T,
    PIXELTYPE		*in,
    PIXELTYPE		*out,
    ANASTRUCT		*skin,
    int			xres,
    int			yres,
    float		xo,
    float		yo,
    float		xf,
    float		yf,
    int			z_interp);

#endif

void
print_anastruct(ANASTRUCT *a,
                int debug,
                int surface,
                int volume);

void
print_date(FILE *stream, PDATE *date);

int generate_tiles(ANASTRUCT *ana, T_STRIPS *tstrips);

void
prism(ANASTRUCT   *obj,
      PNT3D       p1,
      PNT3D       p2,
      int         *inside,
      float       *nearest,
      float       *distance,
      float       *pathlength);

int
prism_ntrsc(
    CONTOUR     *con,
    PNT3D       p1,
    PNT3D       p2,
    float       intersects[2][FI_LIMIT],
    float       int_alpha[FI_LIMIT],
    int         *inside);

#ifdef _PLUNC_HAVE_EXTBM_H
int reconcile_beam(EXT_BEAM *bm, int force);
#endif

void reduce_con(float criterion, CONTOUR *con, int alloc_flag);
float point_to_contour_dist(PNT3D pt, CONTOUR *con);
float point_to_line_segment_dist(PNT3D pt, PNT3D l1, PNT3D l2);

float sample_grid(GRID *grid, PNT3D *pt);
void safe_filename(char *name);

#ifdef _PLUNC_HAVE_PLAN_IM_H
int scale_image(
    int         inres,
    PIXELTYPE   *inptr,
    int         outres,
    PIXELTYPE   *outptr,
    int         min,
    float       scale,
    float       offx,
    float       offy);

int 
scale_image_xy( 
    int         inx,
    int		iny,
    PIXELTYPE   *inptr,
    int         outx,
    int         outy,
    PIXELTYPE   *outptr,
    int         min,
    float       scale,
    float       offx,
    float       offy);
#endif

void spittime(char *str);
double dspittime(char *str);
double dmytime(char *str);
double dcputime(char *str);

int system1(char *command);
int system2(char *command);

int sum_mat(int mcount, float *wts, GRID *nmat, GRID *result);
int sum_vec(int mcount, float *wts, CALC_POINTS *nvec, CALC_POINTS *result);

#if defined(PLAN_LINUX) || defined(PLAN_IRIX) || defined(PLAN_SOLARIS2_4)
#include <unistd.h>
#else
int usleep(unsigned int usec);
#endif

#ifdef PLAN_WINNT
int sleep(int sec);
#endif

float 
v_interp(
    int         mode,
    int         vec_size,
    float       xvec[], 
    float       x, 
    float       yvec[],
    int         *index,
    float       *fx);

float my_erf(float x);

void reset_rand48(int seed);
float grand48(float sigma);
#ifdef PLAN_WINNT
void srand48(int seed);
#endif

void fix_contour_sense(CONTOUR *con, int sense_flag);

/*
#ifdef __cplusplus
}
#endif
*/

#endif

