template<typename Traits>
__global__ void vector_add(float* a, float* b, float* c, int N)  {
    const int block_size = Traits::block_size;

    int idx = blockIdx.x * block_size + threadIdx.x;

    if(idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
