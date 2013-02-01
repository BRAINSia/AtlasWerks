/* ================================================================
 *
 * AtlasWerks Project
 *
 * Copyright (c) Sarang C. Joshi, Bradley C. Davis, J. Samuel Preston,
 * Linh K. Ha. All rights reserved.  See Copyright.txt or for details.
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notice for more information.
 *
 * ================================================================ */

#include <iostream>
#include "Window2D.h"

Window2D
::Window2D( int xPos, int yPos, int width, int height, 
	    char* label ) 
  : IView<float, float>(xPos, yPos, width, height, label)
{
}

Window2D::~Window2D()
{
  std::cout << "[Window2D::~Window2D]" << std::endl;
}

