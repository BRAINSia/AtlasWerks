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

#ifndef __CUDA_SORT_CONFIG_H
#define __CUDA_SORT_CONFIG_H


/*
  WARP_SIZE_LOG = log(WARP_SIZE)
  WARP_PER_BLOCK =  (CTA_SIZE / WARP_SIZE_LOG)
  WARP_PER_BLOCK_LOG = log(WARP_PER_BLOCK)
 */

#define WARP_SIZE             32
#define WARP_SIZE_LOG         5                       
#define CTA_SIZE              256
#define WARP_PER_BLOCK        8
#define WARP_PER_BLOCK_LOG    3
#define KWAY                  16
#define KMASK                 (KWAY - 1)


#endif
