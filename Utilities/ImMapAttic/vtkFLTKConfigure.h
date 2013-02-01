/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 *
 * $Id: vtkFLTKConfigure.h.in,v 1.5 2005/04/15 20:52:59 xpxqx Exp $
 *
 * Copyright (c) 2002 - 2004 Sean McInerney
 * All rights reserved.
 *
 * See Copyright.txt or http://vtkfltk.sourceforge.net/Copyright.html
 * for details.
 *
 *    This software is distributed WITHOUT ANY WARRANTY; without even 
 *    the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
 *    PURPOSE.  See the above copyright notice for more information.
 *
 */
#ifndef VTK_FLTK_CONFIGURE_H_
#  define VTK_FLTK_CONFIGURE_H_

/** \name    vtkFLTKConfigure
 *  \brief   Autodetect OS, compiler, platform, CPU, and development environ.
 * 
 * Here is where system computed values get stored.
 * These values should only change when the target compile platform changes.
 * 
 * \author  Sean McInerney
 * \version $Revision: 1.5 $
 * \date    $Date: 2005/04/15 20:52:59 $
 */

// VTK
#  include "vtkIndent.h"
#  include "vtkSystemIncludes.h"
#  include "vtkSetGet.h"
#  include "vtkTimeStamp.h"
// FLTK
#  include <FL/Enumerations.H>

// Hack for VTK 4.4.0 vtkstd_bool
#  if defined(__cplusplus)
#    include "vtkConfigure.h"

#    if (FL_MAJOR_VERSION==1)
#      if (FL_MINOR_VERSION==0)
#        warning "vtkFLTK version requires FLTK version >= 1.1.0."
#      endif
#    elif (FL_MAJOR_VERSION==2)
#      warning "vtkFLTK version not intended for use with FLTK 2.x."
#    else
#      warning "vtkFLTK version not intended for use with this FLTK version."
#    endif /* FLTK Major Version */

#    if (VTK_MAJOR_VERSION==3)
#      warning "vtkFLTK version requires VTK version >= 4.1.0."
#    elif (VTK_MAJOR_VERSION==4)
#      if (VTK_MINOR_VERSION < 1)
#        warning "vtkFLTK version requires VTK version >= 4.1.0."
#      elif (VTK_MINOR_VERSION < 5)
#        define VTK_FLTK_VTK_4
#      else
#        define VTK_FLTK_VTK_5
#      endif
#    elif (VTK_MAJOR_VERSION==5)
#       define VTK_FLTK_VTK_5
#    else
#      warning "vtkFLTK version not intended for use with this VTK version."
#    endif /* VTK Version */
#  endif /* __cplusplus */

/** vtkFLTK Library Version. */
#  define VTK_FLTK_MAJOR_VERSION        0
#  define VTK_FLTK_MINOR_VERSION        7
#  define VTK_FLTK_MICRO_VERSION        2
#  define VTK_FLTK_PATCH_VERSION        1
#  define VTK_FLTK_VERSION              "0.7.2"
#  define VTK_FLTK_RPM_VERSION          "0.7.2-1"

/*@{*/
/** Generic platform type. */
/* #undef WIN32       */
/* #undef APPLE       */
#define UNIX       1
/*@}*/

/*@{*/
/** Flags for the optional VTK kits used. */
#define VTK_USE_HYBRID
#define VTK_USE_RENDERING
/* #undef VTK_USE_PATENTED */
/* #undef VTK_USE_PARALLEL */
/*@}*/

/*@{*/
/** Force on vtkFLTK debug token if it was on in the build environ. */
/* #undef VTK_FLTK_BUILD_DEBUG  */
#  if defined(_DEBUG)
#    define VTK_FLTK_BUILD_DEBUG  1
#  elif defined(NDEBUG)
#    undef  VTK_FLTK_BUILD_DEBUG
#  endif
/*@}*/

/*@{*/
/** Flag this build as shared or static. */
/* #undef VTK_FLTK_BUILD_SHARED_LIBS OFF */
#  ifdef VTK_FLTK_BUILD_SHARED_LIBS
#    define VTK_FLTK_DLL
#  else
#    define VTK_FLTK_STATIC
#  endif

#  if defined(_MSC_VER) && !defined(VTK_FLTK_STATIC)
#    pragma warning ( disable : 4275 )
#  endif
/*@}*/

/*@{*/
/** Flags for the optional kits built. */
/* #undef VTK_FLTK_BUILD_EXAMPLES ON */
/* #undef VTK_FLTK_BUILD_TCL_WRAPPERS  */
/* #undef VTK_FLTK_BUILD_PYTHON_WRAPPERS  */
/* #undef VTK_FLTK_BUILD_JAVA_WRAPPERS  */
/*@}*/

/**
 * Many functions that previously took or returned float (VTK <= 4.2.5)
 * now take or return double.
 */
#  ifndef vtkFloatingPointType
#    define vtkFloatingPointType vtkFloatingPointType
typedef float vtkFloatingPointType;
#  endif /* vtkFloatingPointType */

/*@{*/
/** Macro dealing with windoze specific type hackery for DLL definitions.
 *
 * These are only for externally visible routines and globals.  For
 * internal routines, just use "extern" for type checking and that
 * will not export internal cross-file or forward-declared symbols.
 * Define a macro for declaring procedures return types. We use this to
 * deal with windoze specific type hackery for DLL definitions. Use
 * VTK_FLTK_EXTERN when the prototype for the method is declared.
 * Use VTK_FLTK_IMPLEMENT for the implementation of the method.
 *
 * Example:
 * \code
 *     VTK_FLTK_EXTERN(void) DoIt(void); // in doit.h
 *
 *     VTK_FLTK_IMPLEMENT(void) DoIt(void) { return; } // in doit.c
 * \endcode
 */
#  if defined(WIN32) && !defined(VTK_FLTK_STATIC)
#    if defined(vtkFLTK_EXPORTS)
#      define VTK_FLTK_EXPORT __declspec( dllexport ) 
#    else
#      define VTK_FLTK_EXPORT __declspec( dllimport ) 
#    endif
#    define VTK_FLTK_EXTERN(_type) extern __declspec( dllexport ) _type
#    define VTK_FLTK_IMPLEMENT(_type) __declspec( dllexport ) _type
#  else
#    define VTK_FLTK_EXPORT
#    define VTK_FLTK_EXTERN(_type) extern _type
#    define VTK_FLTK_IMPLEMENT(_type) _type
#  endif /* ! WIN32 DLL */
/*@}*/

#endif /* VTK_FLTK_CONFIGURE_H_ */
/* 
 * End of: $Id: vtkFLTKConfigure.h.in,v 1.5 2005/04/15 20:52:59 xpxqx Exp $.
 * 
 */
