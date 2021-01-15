// write your code into this file
// your kernels can be implemented directly here, or included
// function solveGPU is a device function: it can allocate memory, call CUDA kernels etc.

#define BLOCK 256
#define LANES 32

__global__ void GPUrest(const int* const __restrict__ contacts, const int* const __restrict__ in, int* const infections, const int n, const int iter, int* const out, int* const rest){
    __shared__ int neighborhood[3][BLOCK+3];
    __shared__ int res[BLOCK+1];
    int tid = threadIdx.x;
    int x = (blockIdx.x * blockDim.x) + tid;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int maxIdx = min(BLOCK, n-(blockIdx.x * blockDim.x));
    int pos = y*n + x;

    res[tid] = 0;

    if(blockIdx.y % 4 == 0 || (blockIdx.y + 1) % 4 == 0 && x < n && y < (n / BLOCK) * 4){
        if(threadIdx.x == 0){
                neighborhood[0][0] = x != 0 && y != 0 ? rest[(y-1)*n + (x-1)] : 0;
                neighborhood[1][0] = x != 0 ? rest[pos-1] : 0;
                neighborhood[2][0] = x != 0 && y < n - 1 ? rest[(y+1)*n + (x-1)] : 0;
            }

            if(threadIdx.x == maxIdx-1){
                if(maxIdx == BLOCK){
                    neighborhood[0][maxIdx + 1] = blockIdx.x < ceil((float)n/BLOCK) - 1 && blockIdx.y != 0 ? rest[(y-1) * n + (x + 1)] : 0;
                    neighborhood[1][maxIdx + 1] = blockIdx.x < ceil((float)n/BLOCK) - 1 ? rest[y * n + (x+1)] : 0;
                    neighborhood[2][maxIdx + 1] = blockIdx.y < n - 1 && blockIdx.x < ceil((float)n/BLOCK) - 1 ? rest[(y+1)*n + (x+1)] : 0;
                }
                else{ // maxIdx is less than BLOCK (ie N == 160)
                    neighborhood[0][maxIdx + 1] = 0;
                    neighborhood[1][maxIdx + 1] = 0;
                    neighborhood[2][maxIdx + 1] = 0;
                }
            }

            neighborhood[0][tid + 1] = blockIdx.y != 0 ? in[(y-1)*n + x] : 0;
            neighborhood[1][tid + 1] = in[pos]; 
            neighborhood[2][tid + 1] = blockIdx.y < n - 1 ? in[(y+1)*n + x] : 0;
    }

    __syncthreads();

    if(x < n && y < (n / BLOCK) * 4){
            int in_pos = neighborhood[1][tid + 1]; 
            if (in_pos > 0) {
                res[tid] = in_pos - 1 == 0 ? -30 : in_pos - 1;
            }

            if (in_pos < 0) {
                res[tid] = in_pos + 1;
            }
            if (in_pos == 0) {
                int infected = 0;
                
                infected += (neighborhood[0][tid] > 0) ? 1 : 0;
                infected += (neighborhood[0][tid + 1] > 0) ? 1 : 0;
                infected += (neighborhood[0][tid + 2] > 0) ? 1 : 0;
                infected += (neighborhood[1][tid] > 0) ? 1 : 0;
                infected += (neighborhood[1][tid + 2] > 0) ? 1 : 0;
                infected += (neighborhood[2][tid] > 0) ? 1 : 0;
                infected += (neighborhood[2][tid + 1] > 0) ? 1 : 0;
                infected += (neighborhood[2][tid + 2] > 0) ? 1 : 0;

                if (infected > contacts[pos]) {
                    res[tid] = 10;
                    atomicAdd(&infections[iter], 1);
                }
            }
        }

        __syncthreads();

        if(x < n && y < (n / BLOCK) * 4){
            if(blockIdx.y % 4 == 0){
                out[x * n + ((blockIdx.y >> 2) * BLOCK)] = res[tid]; 
            }
            if((blockIdx.y + 1) % 4 == 0){
                out[x * n + ((blockIdx.y+1 >> 2) * BLOCK) - 1] = res[tid]; 
            }
            
        }

}

__global__ void GPUiter( const int* const __restrict__ contacts, const int* const __restrict__ in, int* const infections, const int n, const int iter, int* const out, int* const rest){

        __shared__ int restTemp[4][LANES+1];
        __shared__ int neighborhood[3][BLOCK+3];
        __shared__ int res[BLOCK+1];
        int tid = threadIdx.x;
        int x = (blockIdx.x * blockDim.x) + tid;
        int y = (blockIdx.y * blockDim.y) + threadIdx.y;
        int maxIdx = min(BLOCK, n-(blockIdx.x * blockDim.x));
        int pos = y*n + x;

        res[tid] = 0;

        if(x < n && y < n){
            neighborhood[0][tid+1] = y != 0 ? in[(y-1)*n + x] : 0;
            neighborhood[1][tid+1] = in[pos];
            if(threadIdx.x == 0){
                restTemp[0][0] = neighborhood[1][0];
                restTemp[1][0] = neighborhood[1][1];
            }
            if(threadIdx.x == BLOCK - 2){
                restTemp[2][0] = neighborhood[1][threadIdx.x];
                restTemp[3][0] = neighborhood[1][threadIdx.x+1];
            }
        }

        for (int i = 0; i < LANES -1; ++i)
        {
            neighborhood[2][tid+1] = y + i < n - 1 ? in[(y+1)*n + x] : 0;
            if(threadIdx.x == 0){
                restTemp[0][i+1] = neighborhood[1][0];
                restTemp[1][i+1] = neighborhood[1][1];
            }
            if(threadIdx.x == BLOCK - 2){
                restTemp[2][i+1] = neighborhood[1][threadIdx.x];
                restTemp[3][i+1] = neighborhood[1][threadIdx.x+1];
            }

            __syncthreads();

            if(x < n && y < n){
                int in_pos = neighborhood[1][tid + 1];
                if (in_pos > 0) {
                    res[tid] = in_pos - 1 == 0 ? -30 : in_pos - 1;
                }

                if (in_pos < 0) {
                    res[tid] = in_pos + 1;
                }
                if (in_pos == 0) {
                    int infected = 0;
                    
                    infected += (neighborhood[0][tid] > 0) ? 1 : 0;
                    infected += (neighborhood[0][tid + 1] > 0) ? 1 : 0;
                    infected += (neighborhood[0][tid + 2] > 0) ? 1 : 0;
                    infected += (neighborhood[1][tid] > 0) ? 1 : 0;
                    infected += (neighborhood[1][tid + 2] > 0) ? 1 : 0;
                    infected += (neighborhood[2][tid] > 0) ? 1 : 0;
                    infected += (neighborhood[2][tid + 1] > 0) ? 1 : 0;
                    infected += (neighborhood[2][tid + 2] > 0) ? 1 : 0;

                    if (infected > contacts[pos]) {
                        res[tid] = 10;
                    }
                }
            }
            __syncthreads();
            if(x < n && y < n){
                out[pos] = res[tid];
            }

            neighborhood[0][tid+1] = neighborhood[1][tid+1];
            neighborhood[1][tid+1] = neighborhood[2][tid+1];
        }

        for (int i = 0; i < 4; ++i)
        {
            if(tid < LANES){
                rest[(blockIdx.x + 0) * n + (blockIdx.y) * LANES + tid] = restTemp[0][tid];
                rest[(blockIdx.x + 1) * n + (blockIdx.y) * LANES + tid] = restTemp[1][tid];
                rest[(blockIdx.x + 2) * n + (blockIdx.y) * LANES + tid] = restTemp[2][tid];
                rest[(blockIdx.x + 3) * n + (blockIdx.y) * LANES + tid] = restTemp[3][tid];
            }
        }
}

void solveGPU(const int* const contacts, int* const town, int* const infections, const int n, const int iters)
{
    int* in = town;
    int* out;
    int* rest;

    if(cudaMalloc((void**)&out, n * n * sizeof(out[0])) != cudaSuccess){
        fprintf(stderr, "CudaMalloc failed ...\n");
        return;
    }

    if(cudaMalloc((void**)&rest, n * ((n / BLOCK) * 4) * sizeof(out[0])) != cudaSuccess){
        fprintf(stderr, "CudaMalloc failed ...\n");
        return;
    }

    dim3 gridSize;
    dim3 blockSize;

    dim3 gridSizeRest;
    dim3 blockSizeRest;
    // If N is less than block, we reduce the amount of threads per block
    if(n < BLOCK){ 
        gridSize.x = 1;
        gridSize.y = n;
        gridSize.z = 1;

        blockSize.x = n;
        blockSize.y = 1;
        blockSize.z = 1;
    }
    else{
        gridSize.x = ceil((float)n/BLOCK);
        gridSize.y = ceil((float)n/LANES);
        gridSize.z = 1;

        blockSize.x = BLOCK;
        blockSize.y = 1;
        blockSize.z = 1;
    }

    gridSizeRest.x = ceil((float)n/BLOCK);
    gridSizeRest.y = (n / BLOCK) * 4;
    gridSizeRest.z = 1;

    blockSizeRest.x = BLOCK;
    blockSizeRest.y = 1;
    blockSizeRest.z = 1;


    printf("GridSize:  %d %d \n", gridSize.x, gridSize.y);
    printf("BlockSize: %d %d \n", blockSize.x, blockSize.y);
    
    for(int i = 0; i < iters; i++){
        GPUiter<<<gridSize, blockSize>>>(contacts, in, infections, n, i, out, rest);
        GPUrest<<<gridSizeRest, blockSizeRest>>>(contacts, in, infections, n, i, out, rest);

        int* tmp = in;
        in = out;
        out = tmp;
    }

    if (in != town)
    {
        cudaMemcpy(town, in, n * n * sizeof(town[0]), cudaMemcpyDeviceToDevice);
        cudaFree(in);
    }
    else
    {
        cudaFree(out);
    }
}
