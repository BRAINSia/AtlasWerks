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


#include "itkDiscreteGaussianImageFilter.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkOutputWindow.h"
#include "itkTextOutput.h"

using namespace itk;

#include <iostream>
#include <sstream>

#include <math.h>
#include <stdlib.h>

int main(int argc, char** argv)
{

  bool dosphere = false;
  if (argc != 1)
    dosphere = true;

  // Use text output
  itk::TextOutput::Pointer textout = itk::TextOutput::New();
  itk::OutputWindow::SetInstance(textout);

  srand(39280694);

  typedef Image<float, 3> FloatImageType;
  typedef Image<unsigned short, 3> UShortImageType;

  typedef itk::RescaleIntensityImageFilter<FloatImageType, UShortImageType>
    ConverterType;

  typedef ImageFileWriter<UShortImageType> WriterType;

  // Generate images
  FloatImageType::IndexType index;

  FloatImageType::RegionType region;
  FloatImageType::SizeType size;

  size[0] = 128;
  size[1] = 128;
  size[2] = 128;
  region.SetSize(size);

  FloatImageType::SpacingType spacing;
  spacing[0] = 1.0;
  spacing[1] = 1.0;
  spacing[2] = 1.0;

  // Foreground probability
  std::cout << "Generating probabilities..." << std::endl;
  FloatImageType::Pointer fgprob = FloatImageType::New();

  fgprob->SetRegions(region);
  fgprob->Allocate();

  fgprob->SetSpacing(spacing);

  fgprob->FillBuffer(0);

  for (index[0] = 0; index[0] < (int)size[0]; index[0]++)
    for (index[1] = 0; index[1] < (int)size[1]; index[1]++)
      for (index[2] = 0; index[2] < (int)size[2]; index[2]++)
      {
        if (!dosphere)
        {
          if (index[0] < 32 || index[0] > 96)
            continue;
          if (index[1] < 32 || index[1] > 96)
            continue;
          if (index[2] < 32 || index[2] > 96)
            continue;
          fgprob->SetPixel(index, 1.0);
        }
        else
        {
          double dist = 0;
          for (int dim = 0; dim < 3; dim++)
          {
            double d = index[dim] - 64;
            dist += d*d;
          }
          if (dist < 32*32)
            fgprob->SetPixel(index, 1.0);
        }
      }

  // Background probabilities
  FloatImageType::Pointer bgprob = FloatImageType::New();

  bgprob->SetRegions(region);
  bgprob->Allocate();

  bgprob->SetSpacing(spacing);

  bgprob->FillBuffer(0);

  for (index[0] = 0; index[0] < (int)size[0]; index[0]++)
    for (index[1] = 0; index[1] < (int)size[1]; index[1]++)
      for (index[2] = 0; index[2] < (int)size[2]; index[2]++)
      {
        if (!dosphere)
        {
          if ((index[0] >= 32 && index[0] <= 96)
            &&
            (index[1] >= 32 && index[1] <= 96)
            &&
            (index[2] >= 32 && index[2] <= 96))
            continue;
          bgprob->SetPixel(index, 1.0);
        }
        else
        {
          double dist = 0;
          for (int dim = 0; dim < 3; dim++)
          {
            double d = index[dim] - 64;
            dist += d*d;
          }
          if (dist >= 32*32)
            bgprob->SetPixel(index, 1.0);
        }
      }

  // Blur probabilities
  std::cout << "Blurring probabilities..." << std::endl;
  typedef itk::DiscreteGaussianImageFilter<FloatImageType, FloatImageType>
    BlurType;

  BlurType::Pointer blur1 = BlurType::New();
  blur1->SetInput(fgprob);
  blur1->SetVariance(0.25);
  blur1->Update();

  BlurType::Pointer blur2 = BlurType::New();
  blur2->SetInput(bgprob);
  blur2->SetVariance(0.25);
  blur2->Update();

  ConverterType::Pointer converter = ConverterType::New();
  converter->SetOutputMinimum(0);
  converter->SetOutputMaximum(65355);

  WriterType::Pointer writer = WriterType::New();
  writer->UseCompressionOn();

  converter->SetInput(blur1->GetOutput());
  converter->Update();
  writer->SetFileName("p0.gipl");
  writer->SetInput(converter->GetOutput());
  writer->Update();

  converter->SetInput(blur2->GetOutput());
  converter->Update();
  writer->SetFileName("p1.gipl");
  writer->SetInput(converter->GetOutput());
  writer->Update();

  return 0;

}
