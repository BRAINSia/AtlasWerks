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

#ifndef __LIB_DEFINE__H
#define __LIB_DEFINE__H

const int CTA_SIZE=256;
const int CUDA_DATA_BLOCK_ALIGN = 128;

// #define SizeType    Vector3Di
// #define SpacingType Vector3Df
// #define OriginType  Vector3Df

enum BackgroundStrategy { BACKGROUND_STRATEGY_PARTIAL_ID,
                          BACKGROUND_STRATEGY_ID,
                          BACKGROUND_STRATEGY_PARTIAL_ZERO,
                          BACKGROUND_STRATEGY_ZERO,
                          BACKGROUND_STRATEGY_CLAMP,
                          BACKGROUND_STRATEGY_WRAP };


enum mem_type { cpu_mem, gpu_mem, undef_mem};



#endif
