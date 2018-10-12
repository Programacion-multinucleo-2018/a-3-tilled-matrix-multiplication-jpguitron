#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>


using namespace std;
#define N 1000
#define TILE 16
#define TX 16
#define TY 16

void fillMatrix(float * matrix)
{
    int i;
    int size = N*N;
    for(i = 0; i < size; i++)
    {
        matrix[i] = (float)rand()/(RAND_MAX/ 10.0f);
    }

    return;
}

int checkResult(float *hostRef, float *gpuRef)
{
    double epsilon = 0.5;
    bool match = 1;
    int size = N*N;

    for (int i = 0; i < size; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f dif %f\n", hostRef[i], gpuRef[i],hostRef[i] - gpuRef[i]);
            break;
        }
    }

    return match;
}

//Print the matrix
void printMatrix(float * m_r)
{
  int size = N*N;
  int x;
  for(x = 0; x < size; x++)
  {
      if(x%N==0)
      {
        printf("\n");
      }
      printf("%f ", m_r[x]);
  }
}

//multiplication of matrices in cpu
void mulMatrix(float * m_r, float * m1, float * m2)
{
  int x;
  int y;
  int z;
  for(y=0;y<N;y++)
  {
    for(z = 0; z < N; z++)
    {
      for(x = 0; x < N; x++)
      {
          m_r[y*N+z] += m1[x+y*N] * m2[x*N+z];
      }
    }
  }
}

__global__ void mulMatrixGPU2D(float *MatA, float *MatB, float *MatC)
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  float sum = 0;

  if (ix < N && iy < N)
  {
    for(int in =0;in<N;in++)
    {
        sum += MatA[ix*N+in] * MatB[in*N+iy];
    }
    MatC[ix*N+iy]=sum;
  }
}


__global__ void mulMatrixGPUTiles(float* A, float* B, float* C)
{
  float sum = 0;
  
  unsigned int ix = threadIdx.x + TILE * blockIdx.x;
  unsigned int iy = threadIdx.y + TILE * blockIdx.y;

  unsigned int x = threadIdx.x;
  unsigned int y = threadIdx.y;

 
  __shared__ float SharedA[TILE][TILE];
  __shared__ float SharedB[TILE][TILE];

  for(int a = 0; a < TILE;a++)// Se inician en 0 los arreglos
  {
    for(int b = 0; b < TILE; b++)
    {
      SharedA[a][b] = 0.0;
      SharedB[a][b] = 0.0;
    }
  }
  
  for (int a = (TILE + N - 1)/TILE; a >=0; a--) //Recorrer todas las tiles se hace Ceil para asegurar de tener todos los datos, se recorre de forma invertida para conservar los 0s.
    {
      if (a*TILE + x < N && iy < N) //Para que no intente acceder a espacios que no existen de la matriz A
        SharedA[y][x] = A[iy*N + a*TILE + x];

      if (a*TILE + y < N && ix < N)
        SharedB[y][x] = B[(a*TILE + y)*N + ix];

      __syncthreads();

      for (int b = 0; b < TILE; b++)
          sum += SharedA[y][b] * SharedB[b][x];
      
      __syncthreads();
    }
    
    if (ix < N && iy < N)
    {
      C[iy*N+ix] = sum;
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // set up data size of matrix
    int nx = N;
    int ny = N;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_m1, *h_m2, *hostRef, *gpuRef, *gpuRefTiles;
    h_m1 = (float *)malloc(nBytes);
    h_m2 = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    gpuRefTiles = (float *)malloc(nBytes);

    // initialize data at host side

    fillMatrix(h_m1);
    fillMatrix(h_m2);

    memset(hostRef, 0, nBytes);
    memset(gpuRefTiles, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result SAFE_CALLs
    auto start_cpu =  chrono::high_resolution_clock::now();
    mulMatrix(hostRef, h_m1, h_m2);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("sumMatrixOnHost elapsed %f ms\n\n", duration_ms.count());

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_m1, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_m2, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    int dimx = TX;
    int dimy = TY;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    /*******************Normal********************************/
    start_cpu =  chrono::high_resolution_clock::now();
    mulMatrixGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;


    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");
    if(checkResult(hostRef, gpuRef))
      printf("They are equal\n\n");
    else
      printf("They are different\n\n");

    /*******************Tiles********************************/
    start_cpu =  chrono::high_resolution_clock::now();
    mulMatrixGPUTiles<<<grid, block>>>(d_MatA, d_MatB, d_MatC);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;

    printf("sumMatrixOnGPUTiles <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,grid.y,block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRefTiles, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");
    if(checkResult(hostRef, gpuRefTiles))
      printf("They are equal\n\n");
    else
      printf("They are different\n\n");  




    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");


    // free host memory
    free(h_m1);
    free(h_m2);
    free(hostRef);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
