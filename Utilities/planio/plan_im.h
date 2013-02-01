
#ifndef _PLUNC_HAVE_PLAN_IM_H
#define _PLUNC_HAVE_PLAN_IM_H

/*
  This header file describes the image storage format used in the
  PLUNC system.

  The conventions for image geometry are as follows:
  - scans should always be viewed from the foot of the patient. In the
     simple case where the patient is supine and head first, the
     patient's left is on the right side of the image and the
     pixel_to_patient[0][0] is positive. For supine, feet first, the
     patient's left is still on the right side of the image but
     the pixel_to_patient[0][0] is negative.
  - PLanUNC's convention is different than the old PLUNC convention
     where the scans were always (?) viewed from the foot of the
     scanner couch.
  - in *PIXEL COORDINATES* px=0 is at the left and py=0 at the top.
     Pixels are 0-indexed. (You knew that didn't you?)
  - scans are conceived as being collections of *SAMPLES*.  This
     implies that the overall size of the imaged area is (resolution -
     1) * pixel_size.
  - positive x is to the right, positive y is up, positive z is toward
     the foot of the couch.  This is a RIGHT-HANDED system.
  - the x origin is couch center, the y origin is tabletop, the z
     origin is an arbitrary clinically-locatable table position.
  - the "table_height" field is the number of pixels above the bottom
     of the image that the top of the table is seen.
  - In *PIXEL COORDINATES* x=0 is px0=(resolution - 1) / 2.0 *FLOATING
     POINT* - for a scan with an even number of pixels, px0 will have
     a fractional part of 0.5, for instance the x=0 world coordinate
     is found in a 256x256 image at px0=127.5 (that's midway between
     pixels 127 and 128).  In *PIXEL COORDINATES* y=0 is
     py0=(resolution - 1 - table_height).  For example if the table
     height for a 256x256 image is 25, the y=0 world coordinate
     occurs at py0=230.0.  The z=0 world coordinate can have any
     fractional pixel value since it is not constrained to lie in a
     sample plane.
  - the "pixel_to_patient_TM" transform is a 4x4 transform that takes
     pixel coordinates from slice 0 and turns them into patient (world)
     coordinates.  To obtain a transformation matrix for any other slice,
     replace pixel_to_patient_TM[3][2] with the z_position of that slice.
     This assumes we are restricted to table and gantry angles of 0.  If
     we want to be able to handle non-zero values for these fields later
     on, it will be necessary to have a unique pixel_to_patient_TM for
     each slice (i.e., put it in per_scan record).
   - the "first slice", or "slice 0" is defined as that record to which 
     plan_im_header.per_scan[0] points.

  Note that this may not be the same as the scanner's inherent
  conventions.  The transformation is made by some scanner-specific magic
  which lives in the routines which create planning image files from
  tape, floppy, or network.  The original sense of the scan lines is
  thrown away.  The original table angle is demagicated and stored
  here as the table angle for the 'standard scanner' - the counterclockwise
  rotation of the table as viewed from above.  Similarly the gantry
  tilt is rationalized to be the counterclockwise rotation of the
  gantry as viewed from the right side of the machine (from +x in
  patient coordinates).  No magic remains in this specification.

MAJOR NOTE:

  We are currently restricting table and gantry tilt to be zero in
  PLUNC software.  The corresponding fields (with implied
  ability to have different values for each scan) are for future
  expansion and are provided here in the interest of saving our having
  to think this spec through again later.  That is probably wishful
  thinking.

TO DO:

  It would probably pay to create a scanner description file (much like
  the treatment unit description file) that rationalizes the magic and
  further allows users to pick the scanner type from a menu.  This
  will certainly be required if it turns out that a single scanner
  uses different view-from conventions for different types of scans
  (head-cephalic vs. body-caudal for instance).
*/

#define ABSURDLY_LARGE_MAX_SCANS (1000)

/*
   Extended header opcodes. These are four byte integer opcodes.
   The most significant byte is 01 for a PLUNC recognizable opcode
   (other application specific categories can use something else).
   The next two bytes give 65536 possible opcodes. The last byte
   indicates what type of thing follows:
	0x00	no data
	0x01	xdr int
	0x02	xdr float
	0x03	xdr string
	0x04	xdr bytes
	0x05	xdr array
	0xff	unknown
	others will be defines as needed
   The four bytes in the file is an xdr int indicating how many
   bytes there are for the data field, it is the application's
   writer routine to compute this appropriately (for example, a
   xdr string of length 9 would have 4 bytes for the length, and
   then 12 bytes (due to paddind to a four byte multiple) for
   the string with the last 3 bytes being 0's, so the total length
   of the xdr string data would be 16 bytes.
*/

#define PLAN_IM_FIRST_OPCODE	0x01000000
#define PLAN_IM_UNIT_NUMBER	0x01000103
#define PLAN_IM_Y_DIM		0x01000201
#define PLAN_IM_Y_SIZE		0x01000302
#define PLAN_IM_END		0x01ffff00

typedef short PIXELTYPE;

/*
   The "enumerator description" array assigns a unique textual name to
   each enum that should be used across all applications and also
   allows an application to query the list of all enum's of a
   specified type simply by looking through all the elements of the
   array. These are used in user interfaces which need to present
   choices in textual form. This eliminates the problem of finding &
   updating all the applications that duplicate this info as
   hard-coded constants every time the enum's change.

   Make sure the enums and the enum_desc arrays are always
   synchronized.
*/

enum position_list
{
    bogus_position, prone, supine, on_left_side, on_right_side
};

#ifdef PLAN_IM_EXTERN
int position_desc_count = 5;
const char *position_desc[] =
{
  "in a bogus position",
  "prone",
  "supine",
  "on left side",
  "on right side"
};
#else
extern int position_desc_count;
extern char *position_desc[];
#endif

enum entry_list
{
    bogus_entry, feet, head
};

#ifdef PLAN_IM_EXTERN
int entry_desc_count = 3;
const char *entry_desc[] =
{
  "bogus",
  "feet",
  "head"
};
#else
extern int entry_desc_count;
extern char *entry_desc[];
#endif

enum scanners
{
    bogus_scanner, somatom, delta_scanner, ge9800,
    picker, simulix, film_scanner
};

#ifdef PLAN_IM_EXTERN
int scanners_desc_count = 7;
const char *scanners_desc[] =
{
  "bogus scanner",
  "somatom",
  "delta scanner",
  "ge9800",
  "picker",
  "simulix",
  "film scanner"
};
#else
extern int scanners_desc_count;
extern char *scanners_desc[];
#endif

/* ---------------------------------------------------------------- */
enum pixel_types
{
    bogus_pixel, ct_number, scout_view, mri_number
};

#ifdef PLAN_IM_EXTERN
int pixel_desc_count = 4;
const char *pixel_desc[] =
{
  "bogus pixel",
  "CT number",
  "scout view",
  "MRI number"
};
#else
extern int pixel_desc_count;
extern char *pixel_desc[];
#endif

/* ---------------------------------------------------------------- */
typedef struct _per_scan_info
{
    float     z_position;	/* z position of center of slice */
				/* (table position for standard scanner) */
    int       offset_ptrs;	/* pointer to slice (includes header offset) */
    float     gantry_angle;	/* as interpreted for standard scanner */
    float     table_angle;	/* as interpreted for standard scanner */
    int	      scan_number;	/* the scanner's concept */
} per_scan_info;
    
typedef struct _plan_im_header
{
    char      unit_number[20];	/* hospital id # */
    char      patient_name[100];
    PDATE     date;	/* see gen.h */
    char      comment[100];
    enum scanners machine_id;	/* currently of only potential use */
    enum position_list patient_position;	/* patient table position */
    enum entry_list whats_first;	/* what goes first into gantry */
    enum pixel_types pixel_type;
    int       slice_count;
    int       resolution;
    float     pixel_size;	/* absolute value - square assumed */
    int       x_dim;
    int       y_dim;		/* defaults to x_dim=y_dim=resolution */
    float     x_size;
    float     y_size;		/* defaults to x_size=y_size=pixel_size */
    int       table_height;	/* height of table in image (in pixels) */
				/* this will give us the y-origin       */
    int       min;		/* min CT number in study */
    int       max;		/* max CT number in study */
    float     pixel_to_patient_TM[4][4];
				/* transformation matrix for going from
				   pixel to patient (world) coordinates */
    per_scan_info per_scan[ABSURDLY_LARGE_MAX_SCANS];
} plan_im_header;
/* images follow */

/*
  The scans are stored in the file in arbitrary order - probably the
  order in which they are read.  The per_scan array is kept in
  ascending z order.
*/


#endif
