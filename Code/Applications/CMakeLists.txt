## Affine
Mark_AS_ADVANCED(atlasWerks_COMPILE_APP_Affine)
OPTION(atlasWerks_COMPILE_APP_Affine "Compile Affine" ON)
IF(atlasWerks_COMPILE_APP_Affine)
  SUBDIRS(Affine)
ENDIF(atlasWerks_COMPILE_APP_Affine)


## AffineAtlas
Mark_AS_ADVANCED(atlasWerks_COMPILE_APP_AffineAtlas)
OPTION(atlasWerks_COMPILE_APP_AffineAtlas "Compile AffineAtlas" ON)
IF(atlasWerks_COMPILE_APP_AffineAtlas)
  SUBDIRS(AffineAtlas)
ENDIF(atlasWerks_COMPILE_APP_AffineAtlas)


## AtlasWerks
MARK_AS_ADVANCED(atlasWerks_COMPILE_APP_ATLAS_WERKS)
OPTION(atlasWerks_COMPILE_APP_ATLAS_WERKS "Compile AtlasWerks" ON)
IF(atlasWerks_COMPILE_APP_ATLAS_WERKS)
  SUBDIRS(AtlasWerks)
ENDIF(atlasWerks_COMPILE_APP_ATLAS_WERKS)


## VectorAtlasWerks
MARK_AS_ADVANCED(atlasWerks_COMPILE_APP_VECTOR_ATLAS_WERKS)
OPTION(atlasWerks_COMPILE_APP_VECTOR_ATLAS_WERKS "Compile VectorAtlasWerks" ON)
IF(atlasWerks_COMPILE_APP_VECTOR_ATLAS_WERKS)
  SUBDIRS(VectorAtlasWerks)
ENDIF(atlasWerks_COMPILE_APP_VECTOR_ATLAS_WERKS)

## fGrowth
MARK_AS_ADVANCED(atlasWerks_COMPILE_APP_FGROWTH)
OPTION(atlasWerks_COMPILE_APP_FGROWTH "Compile fGrowth" ON)
IF(atlasWerks_COMPILE_APP_FGROWTH)
  SUBDIRS(fGrowth)
ENDIF(atlasWerks_COMPILE_APP_FGROWTH)

## fWarp
MARK_AS_ADVANCED(atlasWerks_COMPILE_APP_FWARP)
OPTION(atlasWerks_COMPILE_APP_FWARP "Compile fWarp" ON)
IF(atlasWerks_COMPILE_APP_FWARP)
  SUBDIRS(fWarp)
ENDIF(atlasWerks_COMPILE_APP_FWARP)

## ImageConvert
#MARK_AS_ADVANCED(atlasWerks_COMPILE_APP_ImageConvert)
OPTION(atlasWerks_COMPILE_APP_ImageConvert "Compile ImageConvert" OFF)
IF(atlasWerks_COMPILE_APP_ImageConvert)
  SUBDIRS(ImageConvert)
ENDIF(atlasWerks_COMPILE_APP_ImageConvert)

## ImMap
IF(atlasWerks_COMPILE_APP_IMMAP)
  #message("COMPILING IMMAP")
  SUBDIRS(ImMap)
ENDIF(atlasWerks_COMPILE_APP_IMMAP)

## LDMM
MARK_AS_ADVANCED(atlasWerks_COMPILE_APP_LDMM)
OPTION(atlasWerks_COMPILE_APP_LDMM "Compile LDMM" ON)
IF(atlasWerks_COMPILE_APP_LDMM)
  SUBDIRS(LDMM)
ENDIF(atlasWerks_COMPILE_APP_LDMM)

## Greedy
MARK_AS_ADVANCED(atlasWerks_COMPILE_APP_GREEDY)
OPTION(atlasWerks_COMPILE_APP_GREEDY "Compile Greedy" ON)
IF(atlasWerks_COMPILE_APP_GREEDY)
  SUBDIRS(Greedy)
ENDIF(atlasWerks_COMPILE_APP_GREEDY)

## txApply
MARK_AS_ADVANCED(atlasWerks_COMPILE_APP_TX_APPLY)
OPTION(atlasWerks_COMPILE_APP_TX_APPLY "Compile txApply" ON)
IF(atlasWerks_COMPILE_APP_TX_APPLY)
  SUBDIRS(txApply)
ENDIF(atlasWerks_COMPILE_APP_TX_APPLY)

## txWerks
MARK_AS_ADVANCED(atlasWerks_COMPILE_APP_TX_WERKS)
OPTION(atlasWerks_COMPILE_APP_TX_WERKS "Compile txWerks" ON)
IF(atlasWerks_COMPILE_APP_TX_WERKS)
  SUBDIRS(txWerks)
ENDIF(atlasWerks_COMPILE_APP_TX_WERKS)

## Utilities
MARK_AS_ADVANCED(atlasWerks_COMPILE_APP_UTILITIES)
OPTION(atlasWerks_COMPILE_APP_UTILITIES "Compile utility applications" ON)
IF(atlasWerks_COMPILE_APP_UTILITIES)
  SUBDIRS(Utilities)
ENDIF(atlasWerks_COMPILE_APP_UTILITIES)

IF(COMPILE_EXPERIMENTAL)
  SUBDIRS(Experimental)
ENDIF(COMPILE_EXPERIMENTAL)

