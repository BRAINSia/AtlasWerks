/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 *
 * $Id: vtkFLTKRenderWindowInteractor.h,v 1.22 2005/04/27 02:41:10 xpxqx Exp $
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
#ifndef VTK_FLTK_RENDER_WINDOW_INTERACTOR_H_
#  define VTK_FLTK_RENDER_WINDOW_INTERACTOR_H_
#  include "vtkFLTKConfigure.h" // vtkFLTK configuration
#  include "vtkRenderWindowInteractor.h"

class vtkCommand;

//BTX
class Fl_Window;
class Fl_VTK_Window;
//ETX

/** \class   vtkFLTKRenderWindowInteractor
 *  \brief   FLTK event driven interface for a RenderWindow.
 * 
 * vtkFLTKRenderWindowInteractor is a convenience object that provides event
 * bindings to common graphics functions. For example, camera and actor
 * functions such as zoom-in/zoom-out, azimuth, roll, and pan. IT is one of
 * the window system specific subclasses of vtkRenderWindowInteractor. Please
 * see vtkRenderWindowInteractor documentation for event bindings.
 * 
 * \author  Sean McInerney
 * \version $Revision: 1.22 $
 * \date    $Date: 2005/04/27 02:41:10 $
 * 
 * \sa
 * vtkRenderWindowInteractor vtkRenderWindow Fl_VTK_Window Fl_Window
 */

class VTK_FLTK_EXPORT vtkFLTKRenderWindowInteractor
  : public vtkRenderWindowInteractor
{
public:
  static vtkFLTKRenderWindowInteractor* New (void);
  vtkTypeRevisionMacro(vtkFLTKRenderWindowInteractor,vtkRenderWindowInteractor);
  void PrintSelf (ostream&, vtkIndent);

  /** Initializes the event handlers. */
  virtual void  Initialize (void);

  /** Add a timeout to the event loop for this instance.
   *
   * \param timerType Indicate if this is first timer set or an update
   *                  as Win32 uses repeating timers, whereas X uses
   *                  One shot more timer. if \c timerType is
   *                  VTKXI_TIMER_FIRST Win32 and X should create timer
   *                  otherwise Win32 should exit and X should perform
   *                  AddTimeOut()
   */
  int           CreateTimer (int timerType);

  /** Returns 1 without ant other effects.
   *
   * \note  Timers automatically expire in X windows.
   */
  int           DestroyTimer (void);

  /** Invoke a timer event for this instance. */
  void          OnTimer (void);

  /** Start the FLTK event loop. */
  virtual void  Start (void);

  //BTX
  /*@{*/
  /** Specify Fl_VTK_Window (FLTK "peer" class widget) to use for interaction.
   * This method is one of a couple steps that are required for
   * setting up a vtkRenderWindowInteractor as a widget inside of another user
   * interface. You do not need to use this method if the render window
   * will be a stand-alone window. This is only used when you want the
   * render window to be a subwindow within a larger user interface.
   * In that case, you must tell the render window what X display id
   * to use (should probably be fl_display), and then ask the render window
   * what depth, visual and colormap it wants (should probably use
   * Fl::gl_visual(), Fl_Gl_Window::mode(), and/or Fl_Gl_Window::can_do()).
   * Then, you must create an FLTK top level window with those settings. Then
   * you can create the rest of your user interface as a child of the
   * top level window you created. Eventually, you will create a Fl_VTK_Window
   * to serve as the rendering window. You must use the SetWidget method to
   * tell this Interactor about that widget.
   */
  virtual void  SetWidget (Fl_VTK_Window* window);
  Fl_VTK_Window* GetWidget (void) const { return this->Top; }
  /*@}*/
  
  /*@{*/
  /** This method will store top level shell (parent) widget for the interactor.
   * This method and the method invocation sequence applies for:
   *     1 vtkRenderWindow-Interactor pair in a nested widget hierarchy
   *     multiple vtkRenderWindow-Interactor pairs in the same top level shell
   * It is not needed for
   *     1 vtkRenderWindow-Interactor pair as direct child of a top level shell
   *     multiple vtkRenderWindow-Interactor pairs, each in its own top level 
   *           shell
   *
   * The method, along with EnterNotify event, changes the keyboard focus among
   * the widgets/vtkRenderWindow(s) so the Interactor(s) can receive the proper
   * keyboard events. The following calls need to be made:
   *     vtkRenderWindow's display ID need to be set to the top level shell's
   *           display ID.
   *     vtkFLTKRenderWindowInteractor's Widget has to be set to the 
   *           vtkRenderWindow's container widget
   *     vtkFLTKRenderWindowInteractor's TopLevel has to be set to the top level
   *           shell widget
   * note that the procedure for setting up render window in a widget needs to
   * be followed.  See vtkRenderWindowInteractor's SetWidget method.
   *
   * If multiple vtkRenderWindow-Interactor pairs in SEPARATE windows are 
   * desired, do not set the display ID (Interactor will create them as 
   * needed.  Alternatively, create and set distinct DisplayID for each 
   * vtkRenderWindow. Using the same display ID without setting the parent 
   * widgets will cause the display to be reinitialized every time an 
   * interactor is initialized), do not set the widgets (so the render 
   * windows would be in their own windows), and do not set TopLevelShell 
   * (each has its own top level shell already).
   */
  virtual void  SetTopLevelShell (Fl_Window* parent);
  Fl_Window* GetTopLevelShell (void) const { return this->TopLevelShell; }
  /*@}*/
  //ETX

  /** Get the mouse position by querying the window server. */
  virtual void  GetMousePosition (int* x, int* y);

  /** Set the command executed on receiving an \c ExitEvent event. */
  void          SetExitObserver (vtkCommand* command);

  /** Reset the exit command to the default. */
  void          SetExitObserverToDefault (void);

protected:
  vtkFLTKRenderWindowInteractor (void);
  ~vtkFLTKRenderWindowInteractor();

  //BTX
  Fl_VTK_Window*        Top;
  Fl_Window*            TopLevelShell;

  int                   OwnTop;

  vtkCommand*           ExitObserver;
  //ETX

  static void   TimerCallback (void*);
  void          Timer (void*); 

private:
  /** \internal Forbidden Default Methods
   *  \note The copy constructor and assignment operator are \e NOT
   *        implemented, thereby forbidding their invokation.
   */
  /*@{*/
  vtkFLTKRenderWindowInteractor (const vtkFLTKRenderWindowInteractor&);
  void operator= (const vtkFLTKRenderWindowInteractor&);
  /*@}*/
};

#endif /* VTK_FLTK_RENDER_WINDOW_INTERACTOR_H_ */
/* 
 * End of: $Id: vtkFLTKRenderWindowInteractor.h,v 1.22 2005/04/27 02:41:10 xpxqx Exp $.
 * 
 */
