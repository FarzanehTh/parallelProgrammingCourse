#include "kernels.h"
#include "filters.h"

__global__ void kernel1(const int8_t *filter, int32_t dimension, 
        const int32_t *input, int32_t *output, int32_t width, int32_t height)
{
        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        if(thread_id >= width * height){
                return;
        }
       /* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */
}

__global__ void normalize1(int32_t *image, int32_t width, int32_t height, int32_t *min_max)
{

        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        if(thread_id >= width * height){
                return;
        }
        /* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */
        normalize_pixel_d(image, pixel_index, smallest, biggest);
}
