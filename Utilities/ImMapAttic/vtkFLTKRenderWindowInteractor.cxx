/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 *
 * $Id: vtkFLTKRenderWindowInteractor.cxx,v 1.28 2005/04/27 02:43:18 xpxqx Exp $
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
#include "vtkFLTKRenderWindowInteractor.h"
// FLTK
#include <FL/Fl.H>
// VTK Common
#include "vtkCommand.h"
#include "vtkObjectFactory.h"
// vtkFLTK
#include "vtkFLTKOpenGLRenderWindow.h"
#include "Fl_VTK_Window.H"


// ----------------------------------------------------------------------------
//      v t k F L T K B r e a k L o o p
// ----------------------------------------------------------------------------
class vtkFLTKBreakLoop : public vtkCommand
{
public:
  static vtkFLTKBreakLoop* New (void) { return new vtkFLTKBreakLoop; }

  void Execute (vtkObject*, unsigned long, void*)
    {
      for (Fl_Window* w = Fl::first_window(); w != NULL; w = Fl::first_window())
        {
        w->hide();
        }
    }

protected:
  vtkFLTKBreakLoop (void) {}
};


// ----------------------------------------------------------------------------
//      v t k F L T K R e n d e r W i n d o w I n t e r a c t o r
// ----------------------------------------------------------------------------
vtkCxxRevisionMacro (vtkFLTKRenderWindowInteractor, "$Revision: 1.28 $");
vtkStandardNewMacro (vtkFLTKRenderWindowInteractor);

// ----------------------------------------------------------------------------
vtkFLTKRenderWindowInteractor::vtkFLTKRenderWindowInteractor (void)
  : Top(NULL),
    TopLevelShell(NULL),
    OwnTop(0),
    ExitObserver(vtkFLTKBreakLoop::New())
{
}

vtkFLTKRenderWindowInteractor::~vtkFLTKRenderWindowInteractor()
{
  this->Disable();

  if (this->Top != NULL)
    {
    if (this->OwnTop)
      {
      delete this->Top;
      }
    this->Top = NULL;
    }

  if (this->ExitObserver != NULL)
    {
    this->ExitObserver->UnRegister(this);
    this->ExitObserver = NULL;
    }
}

// ----------------------------------------------------------------------------
#if 0
void
vtkFLTKRenderWindowInteractor::UnRegister (vtkObjectBase* o)
{
#if 1
  if ( this->RenderWindow != NULL &&
       this->Top != NULL &&
       this->RenderWindow->GetInteractor() == this &&
       this->RenderWindow != o )
    {
    if (this->GetReferenceCount()+this->RenderWindow->GetReferenceCount() == 5)
      {
      Fl_VTK_Window* tmp = this->Top;
      this->RenderWindow->SetInteractor(NULL);
      this->SetWidget(NULL);
      this->SetRenderWindow(NULL);
      delete tmp;
      }
    }
#endif
  this->vtkObject::UnRegister(o);
}
#endif

// ----------------------------------------------------------------------------
void
vtkFLTKRenderWindowInteractor::SetWidget (Fl_VTK_Window* a)
{
  vtkDebugMacro(<< " setting Widget to (Fl_VTK_Window *) " << (void *) a);

  this->Top    = a;
  this->OwnTop = 0;
} 

void
vtkFLTKRenderWindowInteractor::SetTopLevelShell (Fl_Window* a)
{
  vtkDebugMacro(<< " setting TopLevelShell to (Fl_Window *) " << (void *) a);

  this->TopLevelShell = a;
}

// ----------------------------------------------------------------------------
void
vtkFLTKRenderWindowInteractor::Start (void)
{
  vtkDebugMacro(<<"Start() (subclass method)");

  // Let the compositing handle the event loop if it wants to.
  if (this->HasObserver(vtkCommand::StartEvent))
    {
    this->InvokeEvent(vtkCommand::StartEvent,0);
    return;
    }

  if (!this->Initialized)
    {
    this->Initialize();
    }
  if (!this->Initialized)
    {
    return;
    }

  this->AddObserver(vtkCommand::ExitEvent, this->ExitObserver);

  // Start the FLTK event loop.
  //int fl_ret = Fl::run();
  Fl::run();

  this->RemoveObserver(this->ExitObserver);
}

//---------------------------------------------------------------------------
void
vtkFLTKRenderWindowInteractor::Initialize (void)
{
  vtkFLTKOpenGLRenderWindow* renderWindow;
  Fl_Group* parent;
  int* size;
  int* pos;
  
  // Make sure we have a RenderWindow of the expected type.
  if (this->RenderWindow == NULL)
    {
    vtkErrorMacro(<<"Initialize(): No RenderWindow defined!");
    return;
    }
  if ( ( renderWindow =
         vtkFLTKOpenGLRenderWindow::SafeDownCast(this->RenderWindow) ) == NULL )
    {
    vtkErrorMacro(<<"Initialize(): RenderWindow not a vtkFLTK specialization!");
    return;
    }

  // Do initialization stuff.
  this->Initialized = 1;

  size      = renderWindow->GetSize();
  pos       = renderWindow->GetPosition();
  parent    = renderWindow->GetFlParent();

  // Create our own window.
  if ((this->Top = renderWindow->GetFlWindow()) == NULL)
    {
    // NOTE: Fl_Gl_Window makes sure that a modifying call to resize()
    //       is made at the end of its constructor.
    this->Top = new Fl_VTK_Window( this, pos[0], pos[1], size[0], size[1],
                                   renderWindow->GetWindowName() );
    this->OwnTop = 1;

    if (parent != NULL)
      {
      parent->add(static_cast<Fl_Widget*>(this->Top));
      }
    else if (this->TopLevelShell != NULL)
      {
      this->TopLevelShell->add(static_cast<Fl_Widget*>(this->Top));
      }
    }
  // Use a supplied window.
  else
    {
    if (parent != NULL)
      {
      if (parent != this->Top->parent())
        {
        parent->add(static_cast<Fl_Widget*>(this->Top));
        }
      }
    else if (this->TopLevelShell != NULL)
      {
      if (this->TopLevelShell != this->Top->window())
        {
        this->TopLevelShell->add(static_cast<Fl_Widget*>(this->Top));
        }
      }

    if (this->Top->GetInteractor() != this)
      {
      this->Top->SetInteractor(this);
      }

    this->Top->resize_(pos[0], pos[1], size[0], size[1]);
    }

  // Set the parent group.
  if (parent == NULL)
    {
    parent = this->Top->parent();
    renderWindow->SetFlParent(parent);
    }

  // Set the parent window.
  if (this->TopLevelShell == NULL)
    {
    this->SetTopLevelShell(this->Top->window());
    }

  // realize the widget
  if (this->TopLevelShell != NULL)
    this->TopLevelShell->show();
  else
    this->Top->show();
  // Flushes the output buffer and wait until all requests
  // have been received and processed by the server.
  Fl::check();

  //  Find the current window size 
  renderWindow->SetPosition(this->Top->x(), this->Top->y());
  renderWindow->SetSize(this->Top->w(), this->Top->h());

  renderWindow->SetFlWindow(this->Top);

  renderWindow->Start();

  this->Enable();

  this->Size[0] = this->Top->w();
  this->Size[1] = this->Top->h();
}

//---------------------------------------------------------------------------
// FLTK needs global timer callbacks, but we set it up so that this global
// callback knows which instance OnTimer() to call
void
vtkFLTKRenderWindowInteractor::TimerCallback (void* a)
{
  vtkFLTKRenderWindowInteractor* iren;

  if ((iren = reinterpret_cast<vtkFLTKRenderWindowInteractor*>(a)) != NULL)
    {
    iren->OnTimer();
    }
}

int
vtkFLTKRenderWindowInteractor::CreateTimer (int aTimerType) 
{
  // to be called every 10 milliseconds, one shot timer
  // we pass "this" so that the correct OnTimer instance will be called
  if (aTimerType == VTKI_TIMER_FIRST)
    {
    Fl::add_timeout( 0.01,
                     vtkFLTKRenderWindowInteractor::TimerCallback,
                     (void *) this );
    }
  else
    {
    Fl::repeat_timeout( 0.01,
                        vtkFLTKRenderWindowInteractor::TimerCallback,
                        (void *) this );
    }

  return 1;
}

int
vtkFLTKRenderWindowInteractor::DestroyTimer (void) 
{
  // timers automatically expire in X windows ... what about others?
  return 1;
}

void
vtkFLTKRenderWindowInteractor::OnTimer (void) 
{
  if (!this->Enabled)
    {
    return;
    }

  // This is all we need to do, InteractorStyle is stateful and will
  // continue with whatever it's busy
  this->InvokeEvent(vtkCommand::TimerEvent, NULL);
}

void
vtkFLTKRenderWindowInteractor::Timer (void* aPtr)
{
  vtkFLTKRenderWindowInteractor::TimerCallback(aPtr);
}

// ----------------------------------------------------------------------------
void
vtkFLTKRenderWindowInteractor::GetMousePosition (int* aX, int* aY)
{
  Fl::get_mouse(*aX, *aY);

  *aY = this->Size[1] - *aY - 1;
}

// ----------------------------------------------------------------------------
void
vtkFLTKRenderWindowInteractor::SetExitObserver (vtkCommand* aCommand)
{
  if (this->ExitObserver != aCommand)
    {
    if (aCommand == NULL)
      {
      this->SetExitObserverToDefault();
      }
    else
      {
      this->RemoveObserver(this->ExitObserver);
      this->ExitObserver->UnRegister(this);

      vtkDebugMacro(<< this->GetClassName() << " (" << this
                    << "): setting ExitObserver to " << aCommand);

      this->ExitObserver = aCommand;
      this->ExitObserver->Register(this);
      this->Modified();

      if (this->Initialized)
        {
        this->AddObserver(vtkCommand::ExitEvent, this->ExitObserver);
        }
      }
    }
}

void
vtkFLTKRenderWindowInteractor::SetExitObserverToDefault (void)
{
  vtkFLTKBreakLoop* command = vtkFLTKBreakLoop::New();

  this->SetExitObserver(command);

  command->Delete();
}

// ----------------------------------------------------------------------------
void
vtkFLTKRenderWindowInteractor::PrintSelf (ostream& aTarget, vtkIndent aIndent)
{
  this->Superclass::PrintSelf(aTarget,aIndent);

  aTarget << aIndent << "Top:                  "
          << (void *) this->Top << endl;

  aTarget << aIndent << "TopLevelShell:        "
          << (void *) this->TopLevelShell << endl;

  aTarget << aIndent << "OwnTop:               "
          << (this->OwnTop ? "TRUE" : "FALSE") << endl;

  aTarget << aIndent << "ExitObserver:         "
          << this->ExitObserver << endl;

  if (this->ExitObserver != NULL)
    {
    this->ExitObserver->PrintSelf(aTarget, aIndent.GetNextIndent());
    }
}

/* 
 * End of: $Id: vtkFLTKRenderWindowInteractor.cxx,v 1.28 2005/04/27 02:43:18 xpxqx Exp $.
 * 
 */
