// write your code into this file
// your kernels can be implemented directly here, or included
// function solveGPU is a device function: it can allocate memory, call CUDA kernels etc.
#define DEBUGNUMFPU 32
#define BLOCK 128

__global__ void GPUiter(const int* const contacts, const int* const in, int* const infections, const int n, const int iter, int* const out){

        __shared__ int neighborhood[3][BLOCK+3];
        int x = (blockIdx.x * (blockDim.x * 2)) + threadIdx.x;
        int y = blockIdx.y;
        int maxIdx = min(BLOCK, n-(blockIdx.x * (blockDim.x * 2)));
        int pos = y*n + x;
        int offset = maxIdx >> 1;

        if(x + offset < n && y < n){
            if(threadIdx.x == 0){
                neighborhood[0][0] = x != 0 && y != 0 ? in[(y-1)*n + (x-1)] : 0;
                neighborhood[1][0] = x != 0 ? in[pos-1] : 0;
                neighborhood[2][0] = x != 0 && y < n - 1 ? in[(y+1)*n + (x-1)] : 0;
            }

            if(threadIdx.x + offset == maxIdx-1){
                if(maxIdx == BLOCK){
                    neighborhood[0][maxIdx + 1] = blockIdx.x < ceil((float)n/BLOCK) - 1 && blockIdx.y != 0 ? in[(y-1) * n + (x + 1 + offset)] : 0;
                    neighborhood[1][maxIdx + 1] = blockIdx.x < ceil((float)n/BLOCK) - 1 ? in[y * n + (x + 1 + offset)] : 0;
                    neighborhood[2][maxIdx + 1] = blockIdx.y < n - 1 && blockIdx.x < ceil((float)n/BLOCK) - 1 ? in[(y+1)*n + (x + 1 + offset)] : 0;
                }
                else{ // maxIdx is less than BLOCK (ie N == 160)
                    neighborhood[0][maxIdx + 1] = 0;
                    neighborhood[1][maxIdx + 1] = 0;
                    neighborhood[2][maxIdx + 1] = 0;
                }
            }

            neighborhood[0][threadIdx.x + 1] = blockIdx.y != 0 ? in[(y-1)*n + x] : 0;
            neighborhood[0][threadIdx.x + 1 + offset] = blockIdx.y != 0 ? in[(y-1)*n + x + offset] : 0;

            neighborhood[1][threadIdx.x + 1] = in[pos];
            neighborhood[1][threadIdx.x + 1 + offset] = in[pos + offset];

            neighborhood[2][threadIdx.x + 1] = blockIdx.y < n - 1 ? in[(y+1)*n + x] : 0;
            neighborhood[2][threadIdx.x + 1 + offset] = blockIdx.y < n - 1 ? in[(y+1)*n + x + offset] : 0;
        }

        __syncthreads();

        if(x + offset < n && y < n){
            int in_pos = neighborhood[1][threadIdx.x + 1]; 
            if (in_pos > 0) {
                out[pos] = in_pos - 1 == 0 ? -30 : in_pos - 1;
            }

            if (in_pos < 0) {
                out[pos] = in_pos + 1;
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

                if (infected > contacts[pos]) {
                    out[pos] = 10;
                    atomicAdd(&infections[iter], 1);
                }
                else{
                    out[pos] = 0;
                }
            }
  
            int in_pos2 = neighborhood[1][threadIdx.x + 1 + offset];
            if(in_pos2 > 0){
                out[pos + offset] = in_pos2 - 1 == 0 ? -30 : in_pos2 - 1;
            }
            if(in_pos2 < 0){
                out[pos + offset] = in_pos2 + 1; 
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

                if (infected2 > contacts[pos + offset]) {
                    out[pos + offset] = 10;
                    atomicAdd(&infections[iter], 1);
                }
                else{
                    out[pos + offset] = 0;
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

        blockSize.x = n >> 1;
        blockSize.y = 1;
        blockSize.z = 1;
    }
    else{
        gridSize.x = ceil((float)n/BLOCK);
        gridSize.y = n;
        gridSize.z = 1;

        blockSize.x = BLOCK >> 1;
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
