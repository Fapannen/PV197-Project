// write your code into this file
// your kernels can be implemented directly here, or included
// function solveGPU is a device function: it can allocate memory, call CUDA kernels etc.
#define DEBUGNUMFPU 32
#define BLOCK 128

__global__ void GPUiter( const int* const __restrict__ contacts, const int* const __restrict__ in, int* const infections, const int n, const int iter, int* const out){

        __shared__ int neighborhood[3][BLOCK+3];
        int tid = threadIdx.x;
        int x = (blockIdx.x * blockDim.x) + tid;
        int y = blockIdx.y;
        int maxIdx = min(BLOCK, n-(blockIdx.x * blockDim.x));
        int pos = y*n + x;

        // Save these in registers for faster access
        int maxIdxInc = maxIdx + 1;
        int tidInc = tid + 1;

        if(x < n && y < n){
            /*
            if(threadIdx.x == 0){
                neighborhood[0][0] = ;
                neighborhood[1][0] = ;
                neighborhood[2][0] = ;
            }

            if(threadIdx.x == maxIdx-1){
                if(maxIdx == BLOCK){
                    neighborhood[0][maxIdxInc] = ;
                    neighborhood[1][maxIdxInc] = ;
                    neighborhood[2][maxIdxInc] = ;
                }
                else{ // maxIdx is less than BLOCK (ie N == 160)
                    neighborhood[0][maxIdxInc] = 0;
                    neighborhood[1][maxIdxInc] = 0;
                    neighborhood[2][maxIdxInc] = 0;
                }
            }*/

            neighborhood[0][tidInc] = blockIdx.y != 0 ? in[(y-1)*n + x] : 0;
            neighborhood[1][tidInc] = in[pos]; 
            neighborhood[2][tidInc] = blockIdx.y < n - 1 ? in[(y+1)*n + x] : 0;
        }

        __syncthreads();

        if(x < n && y < n){
            int in_pos = neighborhood[1][tidInc]; 
            if (in_pos > 0) {
                out[pos] = in_pos - 1 == 0 ? -30 : in_pos - 1;
            }

            if (in_pos < 0) {
                out[pos] = in_pos + 1;
            }
            if (in_pos == 0) {
                int infected = 0;

                if(tid > 0){
                    infected += (neighborhood[0][tid] > 0) ? 1 : 0;
                    infected += (neighborhood[1][tid] > 0) ? 1 : 0;
                    infected += (neighborhood[2][tid] > 0) ? 1 : 0;
                }

                if(tid + 2 < maxIdxInc){
                    infected += (neighborhood[0][tid + 2] > 0) ? 1 : 0;
                    infected += (neighborhood[1][tid + 2] > 0) ? 1 : 0;
                    infected += (neighborhood[2][tid + 2] > 0) ? 1 : 0;
                }
                
                infected += (neighborhood[0][tid + 1] > 0) ? 1 : 0;
                infected += (neighborhood[2][tid + 1] > 0) ? 1 : 0;
                
                int limit = contacts[pos];

                if (infected > limit) {
                    out[pos] = 10;
                    atomicAdd(&infections[iter], 1);
                }
                else{
                    if(tid == 0 || tid == maxIdx-1){
                        if(infected + 3 <= limit){
                            out[pos] = 0;
                        }
                        else{
                            if(tid == 0){
                                infected += x != 0 && y != 0 ? in[(y-1)*n + (x-1)] > 0 ? 1 : 0 : 0;
                                infected += x != 0 ? in[pos-1] > 0 ? 1 : 0 : 0;
                                infected += x != 0 && y < n - 1 ? in[(y+1)*n + (x-1)] > 0 ? 1 : 0 : 0;
                            }
                            else{
                                if(maxIdx == BLOCK){
                                    infected += blockIdx.x < ceil((float)n/BLOCK) - 1 && blockIdx.y != 0 ? in[(y-1) * n + (x + 1)] > 0 ? 1 : 0 : 0;
                                    infected += blockIdx.x < ceil((float)n/BLOCK) - 1 ? in[y * n + (x+1)] > 0 ? 1 : 0 : 0;
                                    infected += blockIdx.y < n - 1 && blockIdx.x < ceil((float)n/BLOCK) - 1 ? in[(y+1)*n + (x+1)] > 0 ? 1 : 0 : 0;
                                }
                            }
                            if (infected > limit) {
                                out[pos] = 10;
                                atomicAdd(&infections[iter], 1);
                            }
                            else{
                                out[pos] = 0;
                            }
                        }
                    }

                    else{
                        out[pos] = 0;    
                    }
                }
            }
        }
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

        blockSize.x = n;
        blockSize.y = 1;
        blockSize.z = 1;
    }
    else{
        gridSize.x = ceil((float)n/BLOCK);
        gridSize.y = n;
        gridSize.z = 1;

        blockSize.x = BLOCK;
        blockSize.y = 1;
        blockSize.z = 1;
    }

    printf("GridSize:  %d %d \n", gridSize.x, gridSize.y);
    printf("BlockSize: %d %d \n", blockSize.x, blockSize.y);
    
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
