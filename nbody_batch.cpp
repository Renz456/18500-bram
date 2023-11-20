#include "hls_math.h"
#include "nbody.h"

void nbody_batch(float BufP[BATCH_SIZE][5],
                        float BufF[BATCH_SIZE][2],
                        float* particles_tmp,
                        float* temp_tmp) {
                            
    const float time_step = 0.01;
    const float G = 6.67430e-11f;
    const float min_cul_radius = 0.25;
    int curr_index;

    Pi: for (int i = 0; i < (INPUT_LENGTH * 5); i += (BATCH_SIZE * 5)){
        #pragma HLS pipeline off
        
        Load_Batch:for (int p = 0; p < BATCH_SIZE; p++){
            //#pragma HLS unroll
            curr_index = i + p * 5;
            BufP[p][0] = particles_tmp[curr_index];      //x
            BufP[p][1] = particles_tmp[curr_index + 1];  //y
            BufP[p][2] = particles_tmp[curr_index + 2];  //vx
            BufP[p][3] = particles_tmp[curr_index + 3];  //vy
            BufP[p][4] = particles_tmp[curr_index + 4];  //mass
            BufF[p][0] = 0;                     //force_x
            BufF[p][1] = 0;                     //force_y
        }
        
        Pj: for (int j = 0; j < (INPUT_LENGTH * 5); j += 5)
        {
            #pragma HLS pipeline off

            // read particle j
            float xj = particles_tmp[j];
            float yj = particles_tmp[j + 1];
            float massj = particles_tmp[j + 4];
            
            
            BATCH_FORCE: for (int b = 0; b < BATCH_SIZE; b++){
                //#pragma HLS unroll
                // Calculate the distance between the two particles in 2D
                // BufP[b][0] = xi, BufP[b][1] = yi, BufP[b][4] = massi
                // BufF[b][0] = force_x, BufF[b][1] = force_y
                if (i + b * 5 == j){continue;}

                float dx = xj - BufP[b][0];
                float dy = yj - BufP[b][1];
                // fixed_t distance = static_cast<fixed_t>(sqrt(static_cast<float>(dx * dx + dy * dy)));
                float distance = sqrt(dx * dx + dy * dy);
                // float distance = 3.0f;
                //  Define gravitational constant
                //  Calculate the gravitational force in 2D

                if (distance <= min_cul_radius)
                {
                    float force_magnitude = (G * BufP[b][4] * massj) / (distance * distance);
                    // Calculate force components in 2D
                    BufF[b][0] += force_magnitude * (dx / distance); //dependency in accumulation
                    BufF[b][1] += force_magnitude * (dy / distance); //dependency in accumulation
                }
            }
                
        }

        
        Update_Batch: for (int p = 0; p < BATCH_SIZE; p++){
            //#pragma HLS unroll
            curr_index = i + p * 5;
            // Calculate acceleration in 2D
            float ax = BufF[p][0] / BufP[p][4];
            float ay = BufF[p][1] / BufP[p][4];
            // Update velocity in 2D using the calculated acceleration and time step
            temp_tmp[curr_index + 2] = BufP[p][2] + ax * time_step;
            temp_tmp[curr_index + 3] = BufP[p][3] + ay * time_step;
            temp_tmp[curr_index] = BufP[p][0] + temp_tmp[curr_index + 2] * time_step;
            temp_tmp[curr_index + 1] = BufP[p][1] + temp_tmp[curr_index + 3] * time_step;
            temp_tmp[curr_index + 4] = BufP[p][4];
        }
    }
}