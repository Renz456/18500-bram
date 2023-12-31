#pragma once

#include <stdlib.h>
#include <iostream>
#include "ap_fixed.h"

#define INPUT_LENGTH (10000)
#define BATCH_SIZE (8)
#define ITERATIONS (10)

typedef ap_fixed<16, 8> fixed_t;


extern "C"
{
  // void nBodySimulation2D(float *particles);
  void krnl_nbody(float *particles, float *temp, int iterations);
  void nbody_batch(float BufP[BATCH_SIZE][5],
                        float BufF[BATCH_SIZE][2],
                        float* particles_tmp,
                        float* temp_tmp);

}



