// write your code into this file
// your kernels can be implemented directly here, or included
// function solveGPU is a device function: it can allocate memory, call CUDA kernels etc.

__global__ void GPUiter(const int* const contacts, const int* const in, int* const infections, const int n, const int iter, int* const out){

        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        int y = (blockIdx.y * blockDim.y) + threadIdx.y;
        int pos = y*n + x;
        int in_pos = in[pos];

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
            for (int yy = max(y-1, 0); yy <= min(y+1, n-1); yy++)
            for (int xx = max(x-1, 0); xx <= min(x+1, n-1); xx++)
                infected += (in[yy*n + xx] > 0) ? 1 : 0;
            infected -= (in_pos > 0) ? 1 : 0;
            if (infected > contacts[pos]) {
                out[pos] = 10;
                atomicAdd(&infections[iter], 1);
            }
            else
                out[pos] = 0;
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

    dim3 gridSize(n/32 , n/32);
    dim3 blockSize(32 ,32);
    
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
