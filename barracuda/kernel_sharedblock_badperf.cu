// write your code into this file
// your kernels can be implemented directly here, or included
// function solveGPU is a device function: it can allocate memory, call CUDA kernels etc.

#define BLOCK 32

__global__ void GPUiter(const int* const contacts, const int* const in, int* const infections, const int n, const int iter, int* const out){

        __shared__ int neighbourhood[BLOCK+2][BLOCK+2];

        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        int y = (blockIdx.y * blockDim.y) + threadIdx.y;
        int pos = y*n + x;

        if(threadIdx.x == 0 && threadIdx.y == 0){
            int calc = ((y-1)*n) + (x - 1);
            neighbourhood[0][0] = blockIdx.x != 0 && blockIdx.y != 0 ? in[calc] : 0;
        }
        else if(threadIdx.x == 31 && threadIdx.y == 31){
            int calc2 = ((y+1)*n) + (x+1);
            neighbourhood[33][33] = (blockIdx.x < (n / 32) - 1) && (blockIdx.y < (n / 32) - 1) ? in[calc2] : 0;
        }
        else if(threadIdx.x == 0){
            neighbourhood[threadIdx.y+1][0] = blockIdx.x != 0 ? in[pos-1] : 0;
        }
        else if(threadIdx.y == 0){
            int calc3 = ((y-1) * n) + x;
            neighbourhood[threadIdx.y][threadIdx.x + 1] = blockIdx.y != 0 ? in[calc3] : 0;
        }
        else if(threadIdx.x == 31){
            neighbourhood[threadIdx.y + 1][33] = blockIdx.x < (n/32) - 1 ? in[pos+1] : 0;
        }
        else if(threadIdx.y == 31){
            int calc4 = ((y+1)*n) + x;
            neighbourhood[threadIdx.y+2][threadIdx.x+1] = blockIdx.y < (n / 32) - 1 ? in[calc4] : 0;
        }

        neighbourhood[threadIdx.y + 1][threadIdx.x + 1] = in[pos];

        __syncthreads();

        int in_pos = neighbourhood[threadIdx.y+1][threadIdx.x+1];

        if (in_pos > 0) {
            out[pos] = in_pos - 1;
            if (out[pos] == 0)
                out[pos] = -30;
        }
        if (in_pos < 0) {
            out[pos] = in_pos + 1;
        }
        if (in_pos == 0) {
            int infected = 0;
            for (int r = -1; r < 2; ++r)
            {
                for (int c = -1; c < 2; ++c)
                {
                    infected += neighbourhood[(y+1)+r][(x+1)+c] > 0 ? 1 : 0;
                }
            }

            infected -= neighbourhood[y+1][x+1] > 0 ? 1 : 0;

            out[pos] = infected > contacts[pos] ? 10 : 0;
            if(infected > contacts[pos]){
                atomicAdd(&infections[iter], 1);
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

    dim3 gridSize(n/BLOCK , n/BLOCK);
    dim3 blockSize(BLOCK ,BLOCK);
    
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
