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

#ifndef __CPLV_STREAM__H
#define __CPLV_STREAM__H
#include <driver_types.h>

void streamCreate(int nStream);
void streamDestroy();
cudaStream_t getStream(int id);

#define STM_NULL NULL
#define STM_H2D  (getStream(0))
#define STM_D2D  (getStream(1))
#define STM_D2H  (getStream(2))

#endif
