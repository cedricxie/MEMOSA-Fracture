// This file os part of FVM
// Copyright (c) 2012 FVM Authors
// See LICENSE file for terms.

#ifndef _STRUCTUREFIELDS_H_
#define _STRUCTUREFIELDS_H_


#include "Field.h"

struct StructureFields
{
  StructureFields(const string baseName);

  Field deformation;
  Field elasticDeformation;
  Field deformationGradient;
  Field deformationFlux;
  Field eta;
  Field etaold;
  Field eta1;
  Field eta1old;
  Field alpha;
  Field density;
  Field deformationN1;
  Field deformationN2;
  Field deformationN3;
  Field tractionX;
  Field tractionY;
  Field tractionZ;
  Field strainX;
  Field strainY;
  Field strainZ;
  Field distortionX;
  Field distortionY;
  Field distortionZ;
  Field plasticDiagStrain;
  Field devStress;
  Field VMStress;
  Field plasticStrain;
  Field temperature;
  Field bodyForce;
  Field volume0;
  Field creepConstant; 
  Field pfv;
  Field eigenvalue;
  Field eigenvector1;
  Field eigenvector2;
  Field eigenvector3;
  Field reStressXX;
  Field reStressXY;
  Field reStressXZ;
  Field reStressYY;
  Field reStressYZ;
  Field reStressZZ;
};

#endif
