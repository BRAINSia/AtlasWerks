/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 *
 * $Id: vtkFLTKOpenGLRenderWindow.h,v 1.23 2005/04/27 02:47:07 xpxqx Exp $
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
#ifndef VTK_FLTK_OPENGL_RENDER_WINDOW_H_
#  define VTK_FLTK_OPENGL_RENDER_WINDOW_H_
#  include "vtkFLTKConfigure.h" // vtkFLTK configuration
#  include "vtkOpenGLRenderWindow.h"

//BTX
class Fl_Group;
class Fl_VTK_Window;
//ETX

class vtkFLTKOpenGLRenderWindowInternal;

/** \class   vtkFLTKOpenGLRenderWindow
 *  \brief   OpenGL rendering window for FLTK interface.
 * 
 * vtkFLTKOpenGLRenderWindow is a concrete implementation of the abstract
 * class vtkRenderWindow and the vtkOpenGLRenderer interfaces to the
 * OpenGL graphics library. Application programmers should normally use
 * vtkRenderWindow instead of the FLTK-OpenGL specific version.
 * 
 * \author  Sean McInerney
 * \version $Revision: 1.23 $
 * \date    $Date: 2005/04/27 02:47:07 $
 * 
 * \sa
 * vtkOpenGLRenderWindow Fl_VTK_Window Fl_Group
 */

class VTK_FLTK_EXPORT vtkFLTKOpenGLRenderWindow : public vtkOpenGLRenderWindow
{
public:
  static vtkFLTKOpenGLRenderWindow* New (void);
  vtkTypeRevisionMacro (vtkFLTKOpenGLRenderWindow, vtkOpenGLRenderWindow);
  void PrintSelf (ostream&, vtkIndent);

  /** Begin the rendering process. */
  virtual void  Start (void);

  /** End the rendering process and display the image. */
  virtual void  Frame (void);

  /** Create an interactor to control renderers in this window. */
  virtual vtkRenderWindowInteractor*    MakeRenderWindowInteractor (void);

  /** Initialize the window for rendering. */
  virtual void  WindowInitialize (void);

  /** Initialize the rendering window.
   *
   * This will setup all system-specific resources. This method and
   * \c Finalize() must be symmetric and it should be possible to call
   * them multiple times, even changing the underlying window Id
   * in-between. This is what \c WindowRemap() does.
   */
  virtual void  Initialize (void);

  /** "Deinitialize" the rendering window.
   *
   * This will shutdown all system-specific resources. After having
   * called this, it should be possible to destroy a window that was
   * used for a SetWindow() call without any ill effects.
   */
  virtual void  Finalize (void);

  /** Remap the rendering window.
   *
   * This is useful for changing properties that can't normally be
   * changed once the window is up.
   */
  virtual void  WindowRemap (void);

  /** Check to see if a mouse button has been pressed.
   *
   * All other events are ignored by this method.
   * This is a useful check to abort a long render.
   *
   * Ideally, you want to abort the render on any event which causes
   * the \c DesiredUpdateRate to switch from a high-quality rate to a
   * more interactive rate.
   */
  virtual int   GetEventPending (void);

  /** Make this window the current OpenGL context. */
  void          MakeCurrent (void);

  /** Tells if this window is current OpenGL context for the calling thread
      We have just gave this definition so that the code will compile with the newer versions of VTK as this is the new abstract method defined in the newer versions
 */

  virtual bool IsCurrent(void)
 {
	return(false);
 }

  /** If called, allow MakeCurrent() to skip cache-check when called.
   * MakeCurrent() reverts to original behavior of cache-checking
   * on the next render.
   */
  void          SetForceMakeCurrent (void);

  /** Updates window size before calling the superclass \c Render(). */
  void          Render (void);

  /** Get the size of the screen in pixels. */
  int*          GetScreenSize (void);

  /** Get the position in screen coordinates (pixels) of the window. */
  int*          GetPosition (void);

  /*@{*/
  /** Move the window to a new position on the display. */
  void          SetPosition (int x, int y);
  void          SetPosition (int a[2]) {this->SetPosition(a[0], a[1]);}
  /*@}*/

  /** Get the width and height in screen coordinates (pixels) of the window. */
  int*          GetSize (void);

  /*@{*/
  /** Specify the size of the rendering window. */
  void          SetSize (int w, int h);
  void          SetSize (int a[2]) {this->SetSize(a[0], a[1]);}
  /*@}*/

  /*@{*/
  /** Keep track of whether the rendering window has been mapped to screen. */
  void          SetMapped (int a);
  int           GetMapped (void);
  /*@}*/

  /** Set the name of the window. This normally appears at top of the window. */
  void          SetWindowName (const char* name);

  /** Render without displaying the window. */
  void          SetOffScreenRendering (int toggle);

  /*@{*/
  /** Hide or Show the mouse cursor.
   *
   * It is nice to be able to hide the default cursor if you want VTK 
   * to display a 3D cursor instead.
   *
   * \note Set cursor position in window (note that (0,0) is the lower left 
   *       corner).
   */
  void          HideCursor (void);
  void          ShowCursor (void);
  /*@}*/

  /** Change the shape of the cursor. */
  virtual void  SetCurrentCursor (int shape);

  /** Change the window to fill the entire screen. */
  void          SetFullScreen (int);

  /** Set the preferred window size to full screen. */
  void          PrefFullScreen (void);

  /** Toggle whether the window manager border is around the window.
   *
   * The default value is true. Under most X window managers
   * this does not work after the window has been mapped.
   */
  void          SetBorders (int a);

  /** Toggle whether the window will be created in a stereo-capable mode.
   *
   * This method must be called before the window is realized. This method
   * overrides the superclass method since this class can actually check
   * whether the window has been realized yet.
   */
  void          SetStereoCapableWindow (int a);

  /** Get the properties of an ideal rendering window. */
  virtual int   GetDesiredVisualMode (void);

  /** Get report of capabilities for the render window. */
  const char*   ReportCapabilities (void);

  /** Does this render window support OpenGL? 0-false, 1-true. */
  int           SupportsOpenGL (void);

  /** Is this render window using hardware acceleration? 0-false, 1-true. */
  int           IsDirect (void);

  //BTX
  /** Get this RenderWindow's parent FLTK group (if any). */
  Fl_Group*     GetFlParent (void);

  /** Sets the FLTK parent of the window that WILL BE created. */
  void          SetFlParent (Fl_Group* group);
  //ETX

  /** Sets the FLTK parent of the window that WILL BE created (dangerously). */
  void          SetFlParent (void* group);

  //BTX
  /** Get this RenderWindow's FLTK window. */
  Fl_VTK_Window* GetFlWindow (void);

  /** Set this RenderWindow to an existing FLTK window. */
  void          SetFlWindow (Fl_VTK_Window* window);
  //ETX

  /** Set this RenderWindow to an existing FLTK window (dangerously). */
  void          SetFlWindow (void* window);

  /*@{*/
  /**
   * Implementation of \c vtkWindow's system independent methods that are
   * used to help interface to native windowing systems.
   *
   * \note
   * These methods can only be used to set \c Fl_Window subclasses
   * as parent since an \c Fl_Group is never associated with an XID.
   */
  void  SetDisplayId (void* id);
  void  SetWindowId (void* id);
  void  SetParentId (void* id);
  void* GetGenericDisplayId (void);
  void* GetGenericWindowId (void);
  void* GetGenericParentId (void);
  void* GetGenericContext (void);
  void* GetGenericDrawable (void);
  void  SetDisplayInfo (char* id);
  void  SetWindowInfo (char* id);
  void  SetParentInfo (char* id);
  void  SetNextWindowId (void* id);
  void  SetNextWindowInfo (char* id);
  /*@}*/

protected:
  vtkFLTKOpenGLRenderWindow (void);
  ~vtkFLTKOpenGLRenderWindow();

  // WARNING: this was a hack to get ImMap to compile
  // these must have been added to VTK since this package was written
  void CreateAWindow() {};
  void DestroyWindow() {};

  //BTX
  vtkFLTKOpenGLRenderWindowInternal* Internal;

  Fl_Group*             FlParent;
  Fl_VTK_Window*        FlWindow;

  int                   Mode; // FLTK OpenGL window capabilities
  int                   OwnFlWindow;
  int                   ScreenSize[2];
  int                   CursorHidden;
  int                   ForceMakeCurrent;
  int                   UsingHardware;
  char*                 Capabilities;
  //ETX

  /** Set this RenderWindow to use fl_display, maybe opening the display.*/
  virtual void  CheckDisplayConnection (void);

private:
  /** \internal Forbidden Default Methods
   *  \note The copy constructor and assignment operator are \e NOT
   *        implemented, thereby forbidding their invokation.
   */
  /*@{*/
  vtkFLTKOpenGLRenderWindow (const vtkFLTKOpenGLRenderWindow&);
  void operator= (const vtkFLTKOpenGLRenderWindow&);
  /*@}*/
};

#endif /* VTK_FLTK_OPENGL_RENDER_WINDOW_H_ */
/* 
 * End of: $Id: vtkFLTKOpenGLRenderWindow.h,v 1.23 2005/04/27 02:47:07 xpxqx Exp $.
 * 
 */
