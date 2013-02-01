/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 *
 * $Id: Fl_VTK_Window.H,v 1.27 2005/07/29 22:37:58 xpxqx Exp $
 *
 * Copyright (c) 2002 - 2005 Sean McInerney
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
#ifndef Fl_VTK_WINDOW_H_
#  define Fl_VTK_WINDOW_H_
#  include "vtkFLTKConfigure.h"
// FLTK
#  include <FL/Fl_Gl_Window.H>
// The vtkDebugLeaks singleton must be initialized before and
// destroyed after all other VTK object singletons.
#  include "vtkDebugLeaksManager.h"     // Needed for proper singleton init.

// hack alert!!! ... for VTK event handler callback (not currently used)
// all subclasses have type() >= this
#  define VTK_WINDOW_TYPE       (FL_WINDOW+2)
#  define VTK_WINDOW_XCLASS     "vtkfltk, Vtkfltk" 

class vtkProp;
class vtkCamera;
class vtkRenderer;
class vtkRendererCollection;
class vtkRenderWindowInteractor;
class vtkRenderWindow;
class vtkObjectFactory;

/** \class   Fl_VTK_Window
 *  \brief   FLTK event-driven interface to a RenderWindow.
 * 
 * FLTK user interface window peer class to a VTK rendering classes with 
 * additional methods for integrating a FLTK event-driven interface to a VTK 
 * pipeline. Fl_VTK_Window provides methods for tracking modification time, 
 * debugging, printing, and handling events. Fl_VTK_Window also performs 
 * reference counting: objects that are reference counted exist as long as 
 * another object uses them. Once the last reference to a reference counted 
 * object is removed, the object will spontaneously destruct. 
 * 
 * \author  Sean McInerney
 * \version $Revision: 1.27 $
 * \date    $Date: 2005/07/29 22:37:58 $
 * 
 * \sa
 * vtkRenderWindowInteractor vtkRenderWindow Fl_Gl_Window
 */

class VTK_FLTK_EXPORT Fl_VTK_Window : public Fl_Gl_Window
{
  // constructor internals
  void Fl_VTK_Window_ (vtkRenderWindowInteractor* = 0);
  void Fl_VTK_Window_ (vtkRenderWindow*);
  static vtkObjectFactory* Factory;

protected:
  // Fl_Gl_Window overrides
  virtual void flush (void);
  virtual void draw (void);
  virtual int  handle (int event);

  friend class vtkFLTKOpenGLRenderWindow;
  friend class vtkFLTKRenderWindowInteractor;

  /** Resize the FLTK window by calling ancestor method. */
  virtual void resize_ (int x, int y, int w, int h);

public:
  /** Change the size or position of the window. */
  virtual void resize (int x, int y, int w, int h);

  /** Make the displayed window contents current. */
  void Update (void);

  /*@{*/
  /** Set/Get the Interactor. */
  virtual void SetInteractor (vtkRenderWindowInteractor*);
  virtual vtkRenderWindowInteractor* GetInteractor (void);
  /*@}*/

  /*@{*/
  /** Set/Get the RenderWindow. */
  virtual void SetRenderWindow (vtkRenderWindow*);
  virtual vtkRenderWindow* GetRenderWindow (void);
  /*@}*/

  /** Return the Collection of Renderers in the RenderWindow. */
  virtual vtkRendererCollection* GetRenderers (void);

  /** Get the first Renderer held in the RenderWindow's Collection. */
  virtual vtkRenderer* GetDefaultRenderer (void);

  /** Get the active Camera held by the default Renderer. */
  virtual vtkCamera* GetDefaultCamera (void);

  /** Add a Prop to the current Renderer. */
  virtual void AddViewProp (vtkProp* prop);

  /** Remove a Prop from the current Renderer. */
  virtual void RemoveViewProp (vtkProp* prop);

public:
  Fl_VTK_Window (int w = 300, int h = 300, const char* label = "");
  Fl_VTK_Window (int x, int y, int w, int h, const char* label = "");

  ~Fl_VTK_Window();

  /** Get a string describing an event enumeration. */
  static const char* GetEventAsString (int event);

  /** Returns a mode given the desired parameters. */
  static int desired_mode (int& doubleBuffer, int& stereo, int& multiSample);

protected:
  Fl_VTK_Window (vtkRenderWindowInteractor*, int, int, int, int, const char*);
  Fl_VTK_Window (vtkRenderWindow*, int, int, int, int, const char*);

  vtkRenderWindowInteractor*    Interactor;
  vtkRenderWindow*              RenderWindow;

  vtkRenderer*                  QueuedRenderer;

  static void ClassInitialize (void);
  static void ClassFinalize (void);

  friend vtkObjectFactory* FlVtkWindowCheckForVtkFLTKObjectFactory (void);

  friend class Fl_VTK_Window_Init;

private:
  Fl_VTK_Window (const Fl_VTK_Window&);  // Not implemented.
  void operator= (const Fl_VTK_Window&);  // Not implemented.
};

// Utility class to make sure the vtkFLTK Object Factory is
// registered before Fl_VTK_Window is used.
class VTK_FLTK_EXPORT Fl_VTK_Window_Init
{
public:
  Fl_VTK_Window_Init (void);
  ~Fl_VTK_Window_Init();
private:
  static unsigned int Count;
};

// This instance will show up in any translation unit that uses
// Fl_VTK_Window.  It will make sure the vtkFLTK Object Factory is
// registered before Fl_VTK_Window is used.
static Fl_VTK_Window_Init FlVtkWindowInitInstance;

#endif /* Fl_VTK_WINDOW_H_ */
/* 
 * End of: $Id: Fl_VTK_Window.H,v 1.27 2005/07/29 22:37:58 xpxqx Exp $.
 * 
 */
