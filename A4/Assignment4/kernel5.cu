#include "kernels.h"

/* This is your own kernel, you should decide which parameters to add
   here*/

__global__ void kernel5(const int8_t *filter, int32_t dimension,
                       const int32_t *input, int32_t *output, int32_t width, int32_t height)
{
      int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
      int N = height * width;

      for (int i = thread_id; i < N / 2; i += blockDim.x * gridDim.x){
           apply_filter_vectorized_int2_d(input, output, filter, i, dimension, width, height);
      }

      // let the last thread process final elements (if there are any)
      if(thread_id == (blockDim.x * gridDim.x - 1) && N % 2 != 0){
         apply_filter_d(filter, dimension, input, output, width, height, N - 1, N - 1);
      }
}


__global__ void normalize5(int32_t *image, int32_t width, int32_t height, int32_t *min_max){

      int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
      int smallest = min_max[0];
      int biggest = min_max[1];
      int N = height * width;

      for (int i = thread_id; i < N / 4; i += blockDim.x * gridDim.x){
         normalize_pixel_vectorized_int4_d(image, i, smallest, biggest);
      }

      // let the last thread process final elements (if there are any)
      int remainder = N % 4;
      if (thread_id == (blockDim.x * gridDim.x - 1) && remainder != 0) {
         while(remainder) {
            int index = N - remainder;
            remainder--;
            normalize_pixel_d(image, index, smallest, biggest);
         }
      }

}