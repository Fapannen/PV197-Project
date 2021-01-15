// write your code into this file
// your kernels can be implemented directly here, or included
// function solveGPU is a device function: it can allocate memory, call CUDA kernels etc.
#define DEBUGNUMFPU 32
#define BLOCK 128
#define LANES 64   

__global__ void GPUiter(const int* const __restrict__ contacts, const int* const __restrict__ in, int* const infections, const int n, const int iter, int* const out){

        __shared__ int neighborhood[3][BLOCK+3];
        __shared__ int res[BLOCK+1];
        int x = (blockIdx.x * (blockDim.x * 2)) + threadIdx.x;
        int y = blockIdx.y * LANES;
        int maxIdx = min(BLOCK, n-(blockIdx.x * (blockDim.x * 2)));
        int pos = y*n + x;
        int offset = maxIdx >> 1;

        /*
        if(threadIdx.x == 0){
            for (int i = 0; i < 32; ++i)
            {
                for (int j = 0; j < 32; ++j)
                {
                    printf("%d ", in[i*n + j]);
                }
                printf("\n");
            }
        }*/

        if(x + offset < n && y < n){
            if(threadIdx.x == 0){
                neighborhood[0][0] = x != 0 && y != 0 ? in[(y-1)*n + (x-1)] : 0;
                neighborhood[1][0] = x != 0 ? in[pos-1] : 0;
            }

            if(threadIdx.x + offset == maxIdx-1){
                if(maxIdx == BLOCK){
                    neighborhood[0][maxIdx + 1] = blockIdx.x < ceil((float)n/BLOCK) - 1 && blockIdx.y != 0 ? in[(y-1) * n + (x + 1 + offset)] : 0;
                    neighborhood[1][maxIdx + 1] = blockIdx.x < ceil((float)n/BLOCK) - 1 ? in[y * n + (x + 1 + offset)] : 0;
                }
                else{ // maxIdx is less than BLOCK (ie N == 160)
                    neighborhood[0][maxIdx + 1] = 0;
                    neighborhood[1][maxIdx + 1] = 0;
                }
            }

            neighborhood[0][threadIdx.x + 1] = blockIdx.y != 0 ? in[(y-1)*n + x] : 0;
            neighborhood[0][threadIdx.x + 1 + offset] = blockIdx.y != 0 ? in[(y-1)*n + x + offset] : 0;

            neighborhood[1][threadIdx.x + 1] = in[pos];
            neighborhood[1][threadIdx.x + 1 + offset] = in[pos + offset];
        }

        #pragma unroll 32
        for (int i = 0; i < LANES; ++i)
        {
            if(threadIdx.x == 0){
                neighborhood[2][0] = x != 0 && y + i < n - 1 ? in[(y+1+i)*n + (x-1)] : 0;
            }

            if(threadIdx.x + offset == maxIdx-1){
                if(maxIdx == BLOCK){
                    neighborhood[2][maxIdx + 1] = y + i < n - 1 && blockIdx.x < ceil((float)n/BLOCK) - 1 ? in[(y+1+i)*n + (x + 1 + offset)] : 0;
                }
                else{ // maxIdx is less than BLOCK (ie N == 160)
                    neighborhood[2][maxIdx + 1] = 0;
                }
            }
            neighborhood[2][threadIdx.x + 1] = y + i < n - 1 ? in[(y+1+i)*n + x] : 0;
            neighborhood[2][threadIdx.x + 1 + offset] = y + i < n - 1 ? in[(y+1+i)*n + x + offset] : 0;

            __syncthreads();

            if(x + offset < n && y + i < n){
                int in_pos = neighborhood[1][threadIdx.x + 1];
                int in_pos2 = neighborhood[1][threadIdx.x + 1 + offset];
                if (in_pos > 0) {
                    res[threadIdx.x] = in_pos - 1 == 0 ? -30 : in_pos - 1;
                }

                if (in_pos < 0) {
                    res[threadIdx.x] = in_pos + 1;
                }
                if (in_pos == 0) {
                    int infected = 0;
                    
                    infected += (neighborhood[0][threadIdx.x] > 0) ? 1 : 0;
                    infected += (neighborhood[0][threadIdx.x + 1] > 0) ? 1 : 0;
                    infected += (neighborhood[0][threadIdx.x + 2] > 0) ? 1 : 0;
                    infected += (neighborhood[1][threadIdx.x] > 0) ? 1 : 0;
                    infected += (neighborhood[1][threadIdx.x + 2] > 0) ? 1 : 0;
                    infected += (neighborhood[2][threadIdx.x] > 0) ? 1 : 0;
                    infected += (neighborhood[2][threadIdx.x + 1] > 0) ? 1 : 0;
                    infected += (neighborhood[2][threadIdx.x + 2] > 0) ? 1 : 0;

                    if (infected > contacts[(y+i)*n + x]) {
                        res[threadIdx.x] = 10;
                        atomicAdd(&infections[iter], 1);
                    }
                    else{
                        res[threadIdx.x] = 0;
                    }
                }

                if(in_pos2 > 0){
                    res[threadIdx.x + offset] = in_pos2 - 1 == 0 ? -30 : in_pos2 - 1;
                }
                if(in_pos2 < 0){
                    res[threadIdx.x + offset] = in_pos2 + 1; 
                }
                if (in_pos2 == 0) {
                    int infected2 = 0;
                    
                    infected2 += (neighborhood[0][threadIdx.x + offset] > 0) ? 1 : 0;
                    infected2 += (neighborhood[0][threadIdx.x + 1 + offset] > 0) ? 1 : 0;
                    infected2 += (neighborhood[0][threadIdx.x + 2 + offset] > 0) ? 1 : 0;
                    infected2 += (neighborhood[1][threadIdx.x + offset] > 0) ? 1 : 0;
                    infected2 += (neighborhood[1][threadIdx.x + 2 + offset] > 0) ? 1 : 0;
                    infected2 += (neighborhood[2][threadIdx.x + offset] > 0) ? 1 : 0;
                    infected2 += (neighborhood[2][threadIdx.x + 1 + offset] > 0) ? 1 : 0;
                    infected2 += (neighborhood[2][threadIdx.x + 2 + offset] > 0) ? 1 : 0;

                    if (infected2 > contacts[(y+i)*n + x + offset]) {
                        res[threadIdx.x + offset] = 10;
                        atomicAdd(&infections[iter], 1);
                    }
                    else{
                        res[threadIdx.x + offset] = 0;
                    }
                }
            }

             __syncthreads();
            if(x + offset < n && y + i < n){
                out[(y+i)*n + x] = res[threadIdx.x];
                out[(y+i)*n + x + offset] = res[threadIdx.x + offset];
            }

            __syncthreads();
            neighborhood[0][threadIdx.x] = neighborhood[1][threadIdx.x];
            neighborhood[0][threadIdx.x + offset] = neighborhood[1][threadIdx.x + offset];
            neighborhood[1][threadIdx.x] = neighborhood[2][threadIdx.x];
            neighborhood[1][threadIdx.x + offset] = neighborhood[2][threadIdx.x + offset];
            if(threadIdx.x < 3){
                neighborhood[0][threadIdx.x + BLOCK] = neighborhood[1][threadIdx.x + BLOCK];
                neighborhood[1][threadIdx.x + BLOCK] = neighborhood[2][threadIdx.x + BLOCK];
            }
        }

        /*
        if(threadIdx.x == 0){
            for (int i = 0; i < 32; ++i)
            {
                for (int j = 0; j < 32; ++j)
                {
                    printf("%d ", out[i*n + j]);
                }
                printf("\n");
            }
        }*/

        
}

void solveGPU(const int* const contacts, int* const town, int* const infections, const int n, const int iters)
{
    int* in = town;
    int* out;
    if(cudaMalloc((void**)&out, n * n * sizeof(out[0])) != cudaSuccess){
        fprintf(stderr, "CudaMalloc failed ...\n");
        return;
    }

    dim3 gridSize;
    dim3 blockSize;


    // If N is less than block, we reduce the amount of threads per block
    if(n < BLOCK){ 
        gridSize.x = 1;
        gridSize.y = n;
        gridSize.z = 1;

        blockSize.x = n >> 1;
        blockSize.y = 1;
        blockSize.z = 1;
    }
    else{
        gridSize.x = ceil((float)n/BLOCK);
        gridSize.y = ceil((float)n/LANES);
        gridSize.z = 1;

        blockSize.x = BLOCK >> 1;
        blockSize.y = 1;
        blockSize.z = 1;
    }
    
    for(int i = 0; i < iters; i++){
        GPUiter<<<gridSize, blockSize>>>(contacts, in, infections, n, i, out);

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
