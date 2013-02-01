/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 *
 * $Id: vtkFLTKObjectFactory.h,v 1.13 2005/04/27 02:37:54 xpxqx Exp $
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
#ifndef VTK_FLTK_OBJECT_FACTORY_H_
#  define VTK_FLTK_OBJECT_FACTORY_H_
#  include "vtkFLTKConfigure.h" // vtkFLTK configuration
#  include "vtkObjectFactory.h"

/** \class   vtkFLTKObjectFactory
 *  \brief   vtkFLTK object factory.
 * 
 * Object Factory creating properly registered Interactor and RenderWindow 
 * for implementing an event-driven interface with FLTK.
 * 
 * \author  Sean McInerney
 * \version $Revision: 1.13 $
 * \date    $Date: 2005/04/27 02:37:54 $
 * 
 * \sa
 * vtkObjectFactory
 */

class VTK_FLTK_EXPORT vtkFLTKObjectFactory : public vtkObjectFactory
{
public:
  static vtkFLTKObjectFactory* New (void);
  vtkTypeRevisionMacro(vtkFLTKObjectFactory, vtkObjectFactory);

  /** Return the version of VTK that the factory was built with. */
  virtual const char* GetVTKSourceVersion (void);
  /** Return a text description of the factory. */
  virtual const char* GetDescription (void);

protected:
  vtkFLTKObjectFactory (void);

private:
  vtkFLTKObjectFactory (const vtkFLTKObjectFactory&);  // Not implemented.
  void operator= (const vtkFLTKObjectFactory&);  // Not implemented.
};

#endif /* VTK_FLTK_OBJECT_FACTORY_H_ */
/* 
 * End of: $Id: vtkFLTKObjectFactory.h,v 1.13 2005/04/27 02:37:54 xpxqx Exp $.
 * 
 */
