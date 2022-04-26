
#include "kernels.h"
#include "filters.h"

// A separate experiment
/***************************************/
// **** kernel6 uses multiple streams
/***************************************/

__global__ void kernel6(const int8_t *filter, int32_t dimension, const int32_t *input,
                        int32_t *output, int32_t width, int32_t height, int32_t stream, int32_t stream_chunk, int32_t num_stream)
{
   int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
   // last chunk will eaither get less than stream_chunk or equal to stream_chunk
   int my_stream_chunk = (stream == num_stream - 1) ? ((height % stream_chunk == 0 ? stream_chunk :  height % stream_chunk)) : stream_chunk;
   if(thread_id >= my_stream_chunk * width){ // for this you need "your" chunk size
      return;
   }
   int start_offset = stream * stream_chunk * width; // for this you need real chunk size
   int col = (thread_id + start_offset) % width;
   int row = (thread_id + start_offset) / width;
   output[row * width + col] = 0;

   for (int i = max(row - dimension / 2, 0); i < min(row + dimension / 2 + 1, height); i++) {
      for (int j = max(col - dimension / 2, 0); j < min(col + dimension / 2 + 1, width); j++) {
            output[row * width + col] += input[i * width + j] * filter[(i - row + dimension / 2) * dimension + (j - col + dimension / 2)] ;
      }
   }
}


__global__ void normalize6(int32_t *image, int32_t width, int32_t height, int32_t *min_max, int32_t stream, int32_t stream_chunk, int32_t num_stream)
{
   int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
   // last chunk will eaither get less than stream_chunk or equal to stream_chunk
   int my_stream_chunk = (stream == num_stream - 1) ? ((height % stream_chunk == 0 ? stream_chunk :  height % stream_chunk)) : stream_chunk;
   if(thread_id >= my_stream_chunk * width){
      return;
   }
   int smallest = min_max[0];
   int biggest = min_max[1];
   int start_offset = stream * stream_chunk * width;

   int pixel_index = thread_id + start_offset;
   normalize_pixel_d(image, pixel_index, smallest, biggest);
} 

