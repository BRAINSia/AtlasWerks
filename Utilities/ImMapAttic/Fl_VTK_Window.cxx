/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 *
 * $Id: Fl_VTK_Window.cxx,v 1.46 2005/07/29 22:37:58 xpxqx Exp $
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
#include "Fl_VTK_Window.H"
// FLTK
#include <FL/Fl.H>
#include <FL/x.H>
#include <FL/gl.h>
// VTK Common
#include "vtkObjectFactory.h"
#include "vtkObjectFactoryCollection.h"
#include "vtkCommand.h"
#include "vtkProp.h"
// VTK Rendering
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"
// vtkFLTK
#include "vtkFLTKObjectFactory.h"
#include "vtkFLTKOpenGLRenderWindow.h"
#include "vtkFLTKRenderWindowInteractor.h"

// ----------------------------------------------------------------------------
//      F l _ V T K _ W i n d o w
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Start of Fl_VTK_Window static initialization stuff
// ----------------------------------------------------------------------------

// Must NOT be initialized.  Default initialization to zero is necessary.
unsigned int Fl_VTK_Window_Init::Count;
vtkObjectFactory* Fl_VTK_Window::Factory;

vtkObjectFactory*
FlVtkWindowCheckForVtkFLTKObjectFactory (void)
{
  vtkObjectFactory*           factory;
  vtkObjectFactoryCollection* factories;

  if ((factories = vtkObjectFactory::GetRegisteredFactories()) != NULL)
    {
    for ( factories->InitTraversal();
          (factory = factories->GetNextItem()) != NULL; )
      if (factory->IsA("vtkFLTKObjectFactory") != 0)
	return factory;

    // Create and register the specialized object factory which will be used 
    //   to create the specialized Interactor vtkFLTKRenderWindowInteractor
    factory = vtkFLTKObjectFactory::New();
    vtkObjectFactory::RegisterFactory(factory);
    factory->Delete(); // reference retained by vtkObjectFactory
    
    if ((factories = vtkObjectFactory::GetRegisteredFactories()) != NULL)
      {
      for ( factories->InitTraversal();
            (factory = factories->GetNextItem()) != NULL; )
        if (factory->IsA("vtkFLTKObjectFactory") != 0)
          return factory;
      }
    else
      {
      Fl::error("vtkObjectFactory::GetRegisteredFactories() returned NULL!");
      return NULL;
      }
    }
  else
    {
    Fl::error("vtkObjectFactory::GetRegisteredFactories() returned NULL!");
    return NULL;
    }

  Fl::error("Could not find registered vtkFLTKObjectFactory!");
  return NULL;
}


// ----------------------------------------------------------------------------
static inline int
VtkWindowTryForVisualMode (int aDoubleBuffer,
                           int aStereoCapable,
                           int aMultiSamples)
{
  // setup the default stuff 
  int mode = (FL_RGB | FL_DEPTH);

  if (aDoubleBuffer)
    mode |= FL_DOUBLE;

  if (aMultiSamples)
    mode |= FL_MULTISAMPLE;

  if (aStereoCapable)
    mode |= FL_STEREO;

  return (Fl_Gl_Window::can_do(mode) ? mode : 0);
}

int
Fl_VTK_Window::desired_mode (int& aDoubleBuffer,
                             int& aStereoCapable,
                             int& aMultiSamples)
{
  int stereo;
  int multi;
  int m = 0;

  // try every possibility stoping when we find one that works
  for (stereo = aStereoCapable; !m && stereo >= 0; stereo--)
    {
    for (multi = aMultiSamples; !m && multi >= 0; multi--)
      {
      m = VtkWindowTryForVisualMode(aDoubleBuffer, stereo, multi);
      if (m && aStereoCapable && !stereo)
        {
        // requested a stereo capable window but we could not get one
        aStereoCapable = 0;
        }
      }
    }

  for (stereo = aStereoCapable; !m && stereo >= 0; stereo--)
    {
    for (multi = aMultiSamples; !m && multi >= 0; multi--)
      {
      m = VtkWindowTryForVisualMode(!aDoubleBuffer, stereo, multi);
      if (m)
        {
        aDoubleBuffer = !aDoubleBuffer;
        }
      if (m && aStereoCapable && !stereo)
        {
        // requested a stereo capable window but we could not get one
        aStereoCapable = 0;
        }
      }
    }

  return m;
}

// ----------------------------------------------------------------------------
Fl_VTK_Window_Init::Fl_VTK_Window_Init (void)
{
  if (++Count == 1) Fl_VTK_Window::ClassInitialize();
}

Fl_VTK_Window_Init::~Fl_VTK_Window_Init()
{
  if (--Count == 0) Fl_VTK_Window::ClassFinalize();
}

void
Fl_VTK_Window::ClassInitialize (void)
{
  if ((Fl_VTK_Window::Factory=FlVtkWindowCheckForVtkFLTKObjectFactory())==NULL)
    {
    Fl::error("Fl_VTK_Window::ClassInitialize() Failed to get object factory.");
    }

  Fl_VTK_Window::Factory->Register(NULL);       // increment reference count
}

void
Fl_VTK_Window::ClassFinalize (void)
{
  if (Fl_VTK_Window::Factory != NULL)
    {
    Fl_VTK_Window::Factory->UnRegister(NULL);   // decrement reference count
    }
}

// ----------------------------------------------------------------------------
// End of Fl_VTK_Window static initialization stuff
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
void
Fl_VTK_Window::Fl_VTK_Window_ (vtkRenderWindowInteractor* aInteractor)
{
  int stereo = 0;
  int multiSample  = 0;
  int doubleBuffer  = 1;

#if !defined(APPLE)
  if ( fl_display == NULL
#  if defined(UNIX)
       || fl_visual == NULL
#  endif /* !UNIX */
    )
#endif /* APPLE */
    {
    // If successful, XVisualInfo is retained in the global variable 
    // fl_visual. Similarly, an appropriate colormap is retained
    // in fl_colormap.
    // If the display was not already open, it should be now.
    Fl::gl_visual(Fl_VTK_Window::desired_mode(doubleBuffer,stereo,multiSample));
    }

  this->type(VTK_WINDOW_TYPE);
  this->xclass(VTK_WINDOW_XCLASS);

  if (aInteractor == NULL)
    {
    aInteractor = vtkRenderWindowInteractor::New();
    }
  this->SetInteractor(aInteractor);

  vtkRenderWindow* renderWindow;

  if ((renderWindow = this->Interactor->GetRenderWindow()) == NULL)
    {
    renderWindow = vtkRenderWindow::New();
    }
  this->SetRenderWindow(renderWindow);
}

void
Fl_VTK_Window::Fl_VTK_Window_ (vtkRenderWindow* aRenderWindow)
{
  int stereo = 0;
  int multiSample  = 0;
  int doubleBuffer  = 1;

#if !defined(APPLE)
  if ( fl_display == NULL
#  if defined(UNIX)
       || fl_visual == NULL
#  endif /* !UNIX */
    )
#endif /* APPLE */
    {
    // If successful, XVisualInfo is retained in the global variable 
    // fl_visual. Similarly, an appropriate colormap is retained
    // in fl_colormap.
    // If the display was not already open, it should be now.
    Fl::gl_visual(Fl_VTK_Window::desired_mode(doubleBuffer,stereo,multiSample));
    }

  this->type(VTK_WINDOW_TYPE);
  this->xclass(VTK_WINDOW_XCLASS);

  if (aRenderWindow == NULL)
    {
    aRenderWindow = vtkRenderWindow::New();
    }

  vtkRenderWindowInteractor* interactor;

  if ((interactor = aRenderWindow->GetInteractor()) == NULL)
    {
    interactor = vtkRenderWindowInteractor::New();
    interactor->SetRenderWindow(aRenderWindow);
    }
  this->SetInteractor(interactor);
  this->SetRenderWindow(aRenderWindow);
}

// ----------------------------------------------------------------------------
Fl_VTK_Window::Fl_VTK_Window (int w, int h, const char* label)
  : Fl_Gl_Window(w,h,label),
    Interactor(NULL), RenderWindow(NULL), QueuedRenderer(NULL)
{
  this->Fl_VTK_Window_();
}

Fl_VTK_Window::Fl_VTK_Window (int x, int y, int w, int h, const char* label)
  : Fl_Gl_Window(x,y,w,h,label),
    Interactor(NULL), RenderWindow(NULL), QueuedRenderer(NULL)
{
  this->Fl_VTK_Window_();
}

Fl_VTK_Window::Fl_VTK_Window (vtkRenderWindowInteractor* iren,
                              int x, int y, int w, int h, const char* label)
  : Fl_Gl_Window(x,y,w,h,label),
    Interactor(NULL), RenderWindow(NULL), QueuedRenderer(NULL)
{
  this->Fl_VTK_Window_(iren);
}

Fl_VTK_Window::Fl_VTK_Window (vtkRenderWindow* rw,
                              int x, int y, int w, int h, const char* label)
  : Fl_Gl_Window(x,y,w,h,label),
    Interactor(NULL), RenderWindow(NULL), QueuedRenderer(NULL)
{
  this->Fl_VTK_Window_(rw);
}

// ----------------------------------------------------------------------------
Fl_VTK_Window::~Fl_VTK_Window()
{
  // according to the fltk docs, destroying a widget does NOT remove it from
  // its parent, so we have to do that explicitly at destruction
  // (and remember, NEVER delete() an instance of this class)
  if (this->parent() != NULL)
    {
    (static_cast<Fl_Group*>(this->parent()))->
      remove(*(static_cast<Fl_Widget*>(this)));
    }

  // Unestablish ownership (unofficially) by Fl_VTK_Window.
  if (this->Interactor != NULL)
    {
    this->Interactor->UnRegister(NULL);
    this->Interactor = NULL;
    }
  if (this->RenderWindow != NULL)
    {
    this->RenderWindow->UnRegister(NULL);
    this->RenderWindow = NULL;
    }
  if (this->QueuedRenderer != NULL)
    {
    this->QueuedRenderer->UnRegister(NULL);
    this->QueuedRenderer = NULL;
    }
}

// ----------------------------------------------------------------------------
void
Fl_VTK_Window::Update (void)
{
  if (this->RenderWindow != NULL) this->RenderWindow->SetForceMakeCurrent();

  this->redraw();
}

// ----------------------------------------------------------------------------
void
Fl_VTK_Window::SetInteractor (vtkRenderWindowInteractor* aInteractor)
{
  if (this->Interactor == aInteractor)
    {
    return;
    }

  // Decrement the reference count of any previously held Interactor.
  if (this->Interactor != NULL)
    {
    this->Interactor->UnRegister(NULL);
    }

  if ((this->Interactor = aInteractor) != NULL)
    {
    this->Interactor->Register(NULL);   // increment reference count
    
    // Attend to FLTK specializations.
    if (this->Interactor->IsA("vtkFLTKRenderWindowInteractor"))
      {
      (static_cast<vtkFLTKRenderWindowInteractor*>(this->Interactor))->
        SetWidget(this);
      }

    if (this->Interactor->GetRenderWindow() != NULL)
      {
      this->SetRenderWindow(this->Interactor->GetRenderWindow());
      }
    else if (this->RenderWindow != NULL)
      {
      this->Interactor->SetRenderWindow(this->RenderWindow);
      }
    }
}

vtkRenderWindowInteractor*
Fl_VTK_Window::GetInteractor (void)
{
  return this->Interactor;
}

// ----------------------------------------------------------------------------
void
Fl_VTK_Window::SetRenderWindow (vtkRenderWindow* aRenderWindow)
{
  if (this->RenderWindow == aRenderWindow)
    return;
  
  // Decrement the reference count of any previously held RenderWindow.
  if (this->RenderWindow != NULL)
    {
    this->RenderWindow->UnRegister(NULL);
    }

  if ((this->RenderWindow = aRenderWindow) != NULL)
    {
    this->RenderWindow->Register(NULL); // increment reference count

    // Attend to FLTK specializations.
    if (this->RenderWindow->IsA("vtkFLTKOpenGLRenderWindow") != 0)
      {
      (static_cast<vtkFLTKOpenGLRenderWindow*>(this->RenderWindow))->
        SetFlWindow(this);
      this->mode(
        (static_cast<vtkFLTKOpenGLRenderWindow*>(this->RenderWindow))->
        GetDesiredVisualMode() );
      }

    if (this->Interactor != NULL)
      {
      this->RenderWindow->SetInteractor(this->Interactor);
      }
    else if (this->RenderWindow->GetInteractor() != NULL)
      {
      this->SetInteractor(this->RenderWindow->GetInteractor());
      }
    // Force the dimensions of the new RenderWindow to match the widget.
    this->RenderWindow->SetSize(this->w(), this->h());
    }
}

vtkRenderWindow*
Fl_VTK_Window::GetRenderWindow (void)
{
  return this->RenderWindow;
}

// ----------------------------------------------------------------------------
vtkRendererCollection*
Fl_VTK_Window::GetRenderers (void)
{
  vtkRendererCollection* rendererCollection = NULL;
  vtkRenderWindow*       renderWindow;

  if ((renderWindow = this->GetRenderWindow()) != NULL)
    {
    // Add any queued Renderer.
    if (this->QueuedRenderer)
      {
      renderWindow->AddRenderer(this->QueuedRenderer);
      this->QueuedRenderer->UnRegister(NULL);
      this->QueuedRenderer = NULL;
      }

    if ((rendererCollection = renderWindow->GetRenderers()) != NULL)
      {
      if (rendererCollection->GetNumberOfItems() < 1)
        {
	vtkRenderer* renderer = vtkRenderer::New();
	renderWindow->AddRenderer(renderer);
	renderer->Delete();
        }
      }
    else
      {
      Fl::error("GetRenderers() could not get Renderers!");
      }

    }
  else
    {
    Fl::error("GetRenderers() could not get RenderWindow!");
    }

  return rendererCollection;
}

// ----------------------------------------------------------------------------
vtkRenderer*
Fl_VTK_Window::GetDefaultRenderer (void)
{
  vtkRenderer* renderer = NULL;

  // Return the queued Renderer if there is one.
  if (this->QueuedRenderer != NULL)
    {
    renderer = this->QueuedRenderer;
    }
  else
    {
    vtkRenderWindow* renderWindow;

    if ((renderWindow = this->GetRenderWindow()) != NULL)
      {
      vtkRendererCollection* rendererCollection;

      if ((rendererCollection = renderWindow->GetRenderers()) != NULL)
        {
        if (rendererCollection->GetNumberOfItems() < 1)
          {
          renderer = vtkRenderer::New();
          if (renderWindow->GetNeverRendered() != 0)
            {
            this->QueuedRenderer = renderer;
            }
          else
            {
            renderWindow->AddRenderer(renderer);
            renderer->Delete();
            }
          }
        else
          {
          rendererCollection->InitTraversal();
          renderer = rendererCollection->GetNextItem();
          }
        }
      else
        {
        Fl::error("GetDefaultRenderer() failed to get Renderers!");
        }
      }
    else
      {
      Fl::error("GetDefaultRenderer() failed to get RenderWindow!");
      }
    }

  return renderer;
}

// ----------------------------------------------------------------------------
vtkCamera*
Fl_VTK_Window::GetDefaultCamera (void)
{
  vtkCamera*   camera = NULL;
  vtkRenderer* renderer;

  if ((renderer = this->GetDefaultRenderer()) != NULL)
    {
    camera = renderer->GetActiveCamera();
    }

  return camera;
}

// ----------------------------------------------------------------------------
void
Fl_VTK_Window::AddViewProp (vtkProp* aProp)
{
  if (aProp == NULL)
    {
    return;
    }

  vtkRenderer* renderer;

  if ((renderer = this->GetDefaultRenderer()) != NULL)
    {
#ifndef VTK_FLTK_VTK_5
    renderer->AddProp(aProp);
#else
    renderer->AddViewProp(aProp);
#endif
    }
}

void
Fl_VTK_Window::RemoveViewProp (vtkProp* aProp)
{
  if (aProp == NULL)
    {
    return;
    }

  vtkRenderer* renderer;

  if ((renderer = this->GetDefaultRenderer()) != NULL)
    {
#ifndef VTK_FLTK_VTK_5
    renderer->RemoveProp(aProp);
#else
    renderer->RemoveViewProp(aProp);
#endif
    }
}

// ----------------------------------------------------------------------------
// FLTK event handlers
// ----------------------------------------------------------------------------
void
Fl_VTK_Window::flush (void)
{
  if (this->QueuedRenderer != NULL)
    {
    this->RenderWindow->AddRenderer(this->QueuedRenderer);
    this->QueuedRenderer->UnRegister(NULL);
    this->QueuedRenderer = NULL;
    }

  this->draw();

#if 0
  std::cerr << "Fl_VTK_Window::flush()\n"
            << "\t context()\t: " << this->context() << "\n"
            << "\t   valid()\t: " << (this->valid()?"YES":"NO")<< "\n"
            << "\t   shown()\t: " << (this->shown()?"YES":"NO")<< "\n"
            << "\t visible()\t: " << (this->visible()?"YES":"NO")<< "\n"
            << "\t    mode()\t: ( "
            << ( (this->mode()&FL_INDEX) ? "FL_INDEX" :
                 ((this->mode()&FL_RGB8) ? "FL_RGB8" : "FL_RGB") )
            << ((this->mode()&FL_DOUBLE) ? " | FL_DOUBLE" : " | FL_SINGLE")
            << ((this->mode()&FL_ACCUM) ? " | FL_ACCUM" : "")
            << ((this->mode()&FL_ALPHA) ? " | FL_ALPHA" : "")
            << ((this->mode()&FL_DEPTH) ? " | FL_DEPTH" : "")
            << ((this->mode()&FL_STENCIL) ? " | FL_STENCIL" : "")
            << ((this->mode()&FL_MULTISAMPLE) ? " | FL_MULTISAMPLE" : "")
            << ((this->mode()&FL_STEREO) ? " | FL_STEREO" : "")
            << " )\n"
            << "\t  can_do()\t: " << (this->can_do() ? "YES" : "NO")
            << std::endl;
#endif
}

// ----------------------------------------------------------------------------
void
Fl_VTK_Window::draw (void)
{
  if (this->Interactor != NULL)
    this->Interactor->Render();
  else if (this->RenderWindow != NULL)
    this->RenderWindow->Render();
}

// ----------------------------------------------------------------------------
void
Fl_VTK_Window::resize_ (int aX, int aY, int aWidth, int aHeight)
{
  this->Fl_Gl_Window::resize(aX, aY, aWidth, aHeight);
}

void
Fl_VTK_Window::resize (int aX, int aY, int aWidth, int aHeight)
{
  // make sure VTK knows about the new situation
  if (this->Interactor != NULL)
    this->Interactor->UpdateSize(aWidth, aHeight);
  else if (this->RenderWindow != NULL)
    this->RenderWindow->SetSize(aWidth, aHeight);

  // resize the FLTK window by calling ancestor method
  this->resize_(aX, aY, aWidth, aHeight);
}

// ----------------------------------------------------------------------------
const char*
Fl_VTK_Window::GetEventAsString (int aEvent)
{
  switch (aEvent)
    {
    case FL_NO_EVENT:
#if defined(UNIX) && !defined(APPLE)
      if (fl_xevent != NULL)
        {
        switch (fl_xevent->type)
          {
          case KeyPress:         return "NO_EVENT (KeyPress)";
          case KeyRelease:       return "NO_EVENT (KeyRelease)";
          case ButtonPress:      return "NO_EVENT (ButtonPress)";
          case ButtonRelease:    return "NO_EVENT (ButtonRelease)";
          case MotionNotify:     return "NO_EVENT (MotionNotify)";
          case EnterNotify:      return "NO_EVENT (EnterNotify)";
          case LeaveNotify:      return "NO_EVENT (LeaveNotify)";
          case FocusIn:          return "NO_EVENT (FocusIn)";
          case FocusOut:         return "NO_EVENT (FocusOut)";
          case KeymapNotify:     return "NO_EVENT (KeymapNotify)";
          case Expose:           return "NO_EVENT (Expose)";
          case GraphicsExpose:   return "NO_EVENT (GraphicsExpose)";
          case NoExpose:         return "NO_EVENT (NoExpose)";
          case VisibilityNotify: return "NO_EVENT (VisibilityNotify)";
          case CreateNotify:     return "NO_EVENT (CreateNotify)";
          case DestroyNotify:    return "NO_EVENT (DestroyNotify)";
          case UnmapNotify:      return "NO_EVENT (UnmapNotify)";
          case MapNotify:        return "NO_EVENT (MapNotify)";
          case MapRequest:       return "NO_EVENT (MapRequest)";
          case ReparentNotify:   return "NO_EVENT (ReparentNotify)";
          case ConfigureNotify:  return "NO_EVENT (ConfigureNotify)";
          case ConfigureRequest: return "NO_EVENT (ConfigureRequest)";
          case GravityNotify:    return "NO_EVENT (GravityNotify)";
          case ResizeRequest:    return "NO_EVENT (ResizeRequest)";
          case CirculateNotify:  return "NO_EVENT (CirculateNotify)";
          case CirculateRequest: return "NO_EVENT (CirculateRequest)";
          case PropertyNotify:   return "NO_EVENT (PropertyNotify)";
          case SelectionClear:   return "NO_EVENT (SelectionClear)";
          case SelectionRequest: return "NO_EVENT (SelectionRequest)";
          case SelectionNotify:  return "NO_EVENT (SelectionNotify)";
          case ColormapNotify:   return "NO_EVENT (ColormapNotify)";
          case ClientMessage:    return "NO_EVENT (ClientMessage)";
          case MappingNotify:    return "NO_EVENT (MappingNotify)";
          } // switch (fl_xevent->type)
        }
      break;
#else
      return "NO_EVENT";
#endif /* !UNIX */

    case FL_PUSH:               return "PUSH";
    case FL_RELEASE:            return "RELEASE";
    case FL_ENTER:              return "ENTER";
    case FL_LEAVE:              return "LEAVE";
    case FL_DRAG:               return "DRAG";
    case FL_FOCUS:              return "FOCUS";
    case FL_UNFOCUS:            return "UNFOCUS";
    case FL_KEYDOWN:            return "KEYDOWN";
    case FL_KEYUP:              return "KEYUP";
    case FL_CLOSE:              return "CLOSE";
    case FL_MOVE:               return "MOVE";
    case FL_SHORTCUT:           return "SHORTCUT";
    case FL_DEACTIVATE:         return "DEACTIVATE";
    case FL_ACTIVATE:           return "ACTIVATE";
    case FL_HIDE:               return "HIDE";
    case FL_SHOW:               return "SHOW";
    case FL_PASTE:              return "PASTE";
    case FL_SELECTIONCLEAR:     return "SELECTIONCLEAR";
    case FL_MOUSEWHEEL:         return "MOUSEWHEEL";
    case FL_DND_ENTER:          return "DND_ENTER";
    case FL_DND_DRAG:           return "DND_DRAG";
    case FL_DND_LEAVE:          return "DND_LEAVE";
    case FL_DND_RELEASE:        return "DND_RELEASE";
    } // switch (aEvent)

  return "<UnknownEvent>";
}

// ----------------------------------------------------------------------------
#if defined(UNIX) && !defined(APPLE)
static inline int
event_width (void)
{ return (reinterpret_cast<const XConfigureEvent *>(fl_xevent))->width; }

static inline int
event_height (void)
{ return (reinterpret_cast<const XConfigureEvent *>(fl_xevent))->height; }
#endif /* !UNIX */

// ----------------------------------------------------------------------------
int
Fl_VTK_Window::handle (int aEvent)
{
#if !defined(UNIX) || defined(APPLE)
  if (aEvent == FL_NO_EVENT)
    {
    return this->Fl_Gl_Window::handle(aEvent);
    }
#endif /* !UNIX */

#if 0
  if (aEvent != FL_MOVE && aEvent != FL_DRAG)
    {
    // if printing type of event
    std::cerr << "Fl_VTK_Window::handle( " << aEvent << " = "
              << Fl_VTK_Window::GetEventAsString(aEvent) << " )"
              << std::cerr;
    }
#endif

  // vtkInteractorStyle implements the "joystick" style of interaction. That 
  //   is, holding down the mouse keys generates a stream of events that 
  //   cause continuous actions (e.g., rotate, translate, pan, zoom). (The 
  //   class vtkInteractorStyleTrackball implements a grab and move style.) 
  //   The event bindings for this class include the following:
  // 
  //  * Keypress j / Keypress t: toggle between joystick (position 
  //     sensitive) and trackball (motion sensitive) styles. In joystick 
  //     style, motion occurs continuously as long as a mouse button is 
  //     pressed. In trackball style, motion occurs when the mouse button is 
  //     pressed and the mouse pointer moves.
  //  * Keypress c / Keypress o: toggle between camera and object (actor) 
  //     modes. In camera mode, mouse events affect the camera position and 
  //     focal point. In object mode, mouse events affect the actor that is 
  //     under the mouse pointer.
  //  * Button 1: rotate the camera around its focal point (if camera mode) 
  //     or rotate the actor around its origin (if actor mode). The rotation 
  //     is in the direction defined from the center of the renderer's 
  //     viewport towards the mouse position. In joystick mode, the magnitude 
  //     of the rotation is determined by the distance the mouse is from the 
  //     center of the render window.
  //  * Button 2: pan the camera (if camera mode) or translate the actor (if 
  //     object mode). In joystick mode, the direction of pan or translation 
  //     is from the center of the viewport towards the mouse position. In 
  //     trackball mode, the direction of motion is the direction the mouse 
  //     moves. (Note: with 2-button mice, pan is defined as <Shift>-Button 1.)
  //  * Button 3: zoom the camera (if camera mode) or scale the actor (if 
  //     object mode). Zoom in/increase scale if the mouse position is in the 
  //     top half of the viewport; zoom out/decrease scale if the mouse 
  //     position is in the bottom half. In joystick mode, the amount of zoom 
  //     is controlled by the distance of the mouse pointer from the 
  //     horizontal centerline of the window.
  //  * Keypress 3: toggle the render window into and out of stereo mode. By 
  //     default, red-blue stereo pairs are created. Some systems support 
  //     Crystal Eyes LCD stereo glasses; you have to invoke 
  //     SetStereoTypeToCrystalEyes() on the rendering window.
  //  * Keypress e: exit the application.
  //  * Keypress p: perform a pick operation. The render window interactor 
  //     has an internal instance of vtkCellPicker that it uses to pick.
  //  * Keypress r: reset the camera view along the current view direction. 
  //     Centers the actors and moves the camera so that all actors are visible.
  //  * Keypress s: modify the representation of all actors so that they 
  //     are surfaces.
  //  * Keypress u: invoke the user-defined function. Typically, this 
  //     keypress will bring up an interactor that you can type commands in.
  //  * Keypress w: modify the representation of all actors so that they 
  //     are wireframe.
  //

  vtkRenderWindowInteractor* interactor;
  vtkRenderWindow*           renderWindow;

  if ( ((interactor   = this->GetInteractor()) == NULL) ||
       ((renderWindow = this->GetRenderWindow()) == NULL) ||
       (this->shown() == 0) )
    {
    return this->Fl_Gl_Window::handle(aEvent);
    }

  int   enabled = interactor->GetEnabled();;

  int   ctrl    = Fl::event_state(FL_CTRL);
  int   shift   = Fl::event_state(FL_SHIFT);

  int   ex      = Fl::event_x();
  int   ey      = Fl::event_y();

  switch (aEvent)
    {

    // 
    // Focus Events ( FOCUS || UNFOCUS || ENTER || LEAVE )
    // 

    case FL_FOCUS:
      // Indicates an attempt to give a widget the keyboard focus (FocusIn).
      // Returning non-zero from handle() means that the widget wants focus.
      // It then becomes the Fl::focus() widget and gets KEYDOWN, 
      // KEYUP, and UNFOCUS events.
      if (enabled)
        {
        return 1; // FOCUS
        }
      break;

    case FL_UNFOCUS:
      // This event is sent to the previous Fl::focus() widget when 
      // another widget gets the focus or the window loses focus (FocusOut).
      if (enabled)
        {
        return 1; // UNFOCUS
        }
      break;

    case FL_ENTER:
      // The mouse has been moved to point at this widget (EnterNotify).
      // Indicate that this wants an FOCUS sent to it.
      {
      if (this->window() != NULL)
        {
        // Sends this window a FOCUS event. It will return 1 (see above)
        // affirming its desire to become the Fl::focus() widget receiving
        // subsequent KEYBOARD events.
        this->take_focus();
        }
      if (enabled)
        {
        interactor->SetEventInformationFlipY(ex, ey, ctrl, shift);
        interactor->InvokeEvent(vtkCommand::EnterEvent, NULL);
        // Returning non-zero indicates that we wish to track the mouse. 
        // This then becomes the Fl::belowmouse() widget and will receive 
        // MOVE and LEAVE events. 
        return 1; // ENTER
        }
      } break;

    case FL_LEAVE:
      // The mouse has moved out of the widget (LeaveNotify). 
      if (enabled)
        {
        interactor->SetEventInformationFlipY(ex, ey, ctrl, shift);
        interactor->InvokeEvent(vtkCommand::LeaveEvent, NULL);
        return 1; // LEAVE
        }
      break;

    // 
    // Widget Events ( SHOW || HIDE || ACTIVATE || DEACTIVATE )
    // 

    case FL_SHOW:
      // This widget is visible again, due to show() being called on it or one
      // of its parents, or due to a parent window being restored (MapNotify). 
      {
      // By not checking to see if the interactor is enabled here,
      // performing the Render() will end up enabling the interactor.
      // This behavior may be undesirable.
      if (renderWindow->GetNeverRendered())
        {
        renderWindow->Render();
        }
      // Child Fl_Windows respond to SHOW by actually creating the window 
      // if not done already, so if you subclass a window, be sure to pass
      // SHOW to the base class handle() method! 
      } break;

#if 0
    case FL_HIDE:
      // Widget is no longer visible, due to hide() being called on it or one of
      // its parents, or due to a parent window being minimized (UnmapNotify).
      // visible() may still be true after this, but the widget is visible only 
      // if visible() is true for it and all its parents (use visible_r() to
      // check this).
      break; // HIDE

    case FL_ACTIVATE:
      // This widget is now active, due to activate() being called on it or 
      // one of its parents. 
      break; // ACTIVATE

    case FL_DEACTIVATE:
      // This widget is no longer active, due to deactivate() being called 
      // on it or one of its parents. active() may still be true after this, 
      // the widget is only active if active() is true on it and all its 
      // parents (use active_r() to check this).
      break; // DEACTIVATE
#endif /* 0 */

    // 
    // Mouse Events ( PUSH || RELEASE || DRAG || MOVE )
    // 

    case FL_PUSH:
      // A mouse button has gone down over this widget (ButtonPress).
      if (enabled)
        {
        interactor->SetEventInformationFlipY(ex, ey, ctrl, shift);
      
        switch (Fl::event_button())
          {
          case FL_LEFT_MOUSE:
            interactor->InvokeEvent(vtkCommand::LeftButtonPressEvent, NULL);
            break;
          case FL_MIDDLE_MOUSE:
            interactor->InvokeEvent(vtkCommand::MiddleButtonPressEvent, NULL);
            break;
          case FL_RIGHT_MOUSE:
            interactor->InvokeEvent(vtkCommand::RightButtonPressEvent, NULL);
            break;
          } // switch (button)
      
        // Indicate that we "want" the mouse click by returning non-zero. 
        // This will then become the Fl::pushed() widget and will get 
        // DRAG and the matching RELEASE events. 
        return 1;
        }
      break; // PUSH
      
    case FL_RELEASE:
      // A mouse button has been released (ButtonRelease).
      if (enabled)
        {
        interactor->SetEventInformationFlipY(ex, ey, ctrl, shift);

        switch (Fl::event_button())
          {
          case FL_LEFT_MOUSE:
            interactor->InvokeEvent(vtkCommand::LeftButtonReleaseEvent, NULL);
            break;
          case FL_MIDDLE_MOUSE:
            interactor->InvokeEvent(vtkCommand::MiddleButtonReleaseEvent, NULL);
            break;
          case FL_RIGHT_MOUSE:
            interactor->InvokeEvent(vtkCommand::RightButtonReleaseEvent, NULL);
            break;
          } // switch (button)
      
        return 1;
        }
      break; // RELEASE

    // we test for both of these, as fltk classifies mouse moves as with or
    // without button press whereas vtk wants all mouse movement (this bug took
    // a while to find :) -cpbotha
    case FL_DRAG:
      // The mouse has moved with a button held down.
    case FL_MOVE:
      // The mouse has moved without any mouse buttons held down (MotionNotify).
      if (enabled)
        {
        interactor->SetEventInformationFlipY(ex, ey, ctrl, shift);
        interactor->InvokeEvent(vtkCommand::MouseMoveEvent, NULL);
        return 1;
        }
      break; // DRAG || MOVE

    // 
    // Keyboard Events
    // 

    // now for possible controversy: there is no way to find out if the 
    // InteractorStyle actually did something with this event. To play it 
    // safe (and have working hotkeys), we return "0", which indicates to 
    // FLTK that we did NOTHING with this event.  FLTK will send this 
    // keyboard event to other children in our group, meaning it should 
    // reach any FLTK keyboard callbacks (including hotkeys)

    case FL_SHORTCUT:
      // If the Fl::focus() widget is zero or ignores an KEYBOARD event 
      // then FLTK tries sending this event to every widget it can, until 
      // one of them returns non-zero. SHORTCUT is first sent to the 
      // Fl::belowmouse() widget, then its parents and siblings, and 
      // eventually to every widget in the window, trying to find an object 
      // that returns non-zero. FLTK tries really hard to not to ignore 
      // any keystrokes! ... FALLING THROUGH ...
    
    // The key can be found in Fl::event_key(). The text that the key should 
    // insert can be found with Fl::event_text() and its length is in 
    // Fl::event_length(). If you use the key, handle() should return 1. 
    // If you return zero, FLTK assumes that you ignored the key and will 
    // attempt to send it to a parent widget. If none of them want it, it 
    // will change the event into a SHORTCUT event. 
    // To receive KEYBOARD events you must also respond to the FOCUS 
    // and UNFOCUS events. 

    case FL_KEYDOWN:
      // A key was pressed (KeyPress).
      if (enabled)
        {
        interactor->SetEventInformationFlipY( ex,ey, ctrl,shift,
                                              Fl::event_key(), 1,
                                              Fl::event_text() );
        interactor->InvokeEvent(vtkCommand::KeyPressEvent, NULL);
        interactor->InvokeEvent(vtkCommand::CharEvent, NULL);
        return 0;
        }
      break; // KEYDOWN
    
    case FL_KEYUP:
      // A key was released (KeyRelease).
      if (enabled)
        {
        interactor->SetEventInformationFlipY( ex,ey, ctrl,shift,
                                              Fl::event_key(), 1,
                                              Fl::event_text() );
        interactor->InvokeEvent(vtkCommand::KeyReleaseEvent, NULL);
        return 0;
        }
      break; // KEYUP

#if 0
#  if defined(UNIX) && !defined(APPLE)
    case FL_NO_EVENT:
      //
      // Other X Events not specifically enumerated by FLTK.
      //
      if (fl_xevent != NULL)
        {
        switch (fl_xevent->type)
          {

          case Expose:
            // ignore child windows
            if (this->parent() == NULL)
              {
              interactor->SetEventSize(event_width(), event_height());
              interactor->SetEventPositionFlipY(ex, ey);

              if (enabled)
                {
                // only render if we are currently accepting events
                interactor->InvokeEvent(vtkCommand::ExposeEvent, NULL);
                renderWindow->Render();
                }
              }
            break; // Expose

          case ConfigureNotify: 
            // ignore child windows
            if (this->parent() == NULL)
              {
              if (event_width() != this->w() || event_height() != this->h())
                {
                interactor->UpdateSize(event_width(), event_height());
                interactor->SetEventPositionFlipY(ex, ey);

                if (enabled)
                  {
                  // only render if we are currently accepting events
                  interactor->InvokeEvent(vtkCommand::ConfigureEvent, NULL);
                  renderWindow->Render();
                  }
                }
              }
            break; // ConfigureNotify

          case EnterNotify:
            break; // EnterNotify

          case LeaveNotify:
            break; // LeaveNotify

          } // switch (fl_xevent->type)
        }
      break; // NO_EVENT
#  endif /* !UNIX */
#endif /* 0 */
    
    } // switch (aEvent)

  //
  // let the base class handle everything else 
  //
  return this->Fl_Gl_Window::handle(aEvent);
}

/* 
 * End of: $Id: Fl_VTK_Window.cxx,v 1.46 2005/07/29 22:37:58 xpxqx Exp $.
 * 
 */
