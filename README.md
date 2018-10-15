# Assignment 3: Tilled Matrix Multiplication

Assignment No 3 for the multi-core programming course. Modify previous matrix multiplication kernels to integrate a tilled multiplication using shared memory.

The program has to do the following:

1. Multiply 2 NxN matrices. N has to be set to 2000. Perform the multiplication with and without tilling.
2. Fill the matrices with random floats between 1 and 10.
3. Validate that the result from the matrix multiplication in GPU with a CPU version. The CPU version does not have to be tilled.
4. Compare the processing time of the matrix multiplication in GPU with and without tilling, and report the speedup obtained.

Execute the kernel at least 20 times, and measure average time spent for calculating the matrix multiplication, and report both the processing times and the speedups within the readme. Test performance varying the number of threads, and the tile window. Test with the following sizes: 8x8, 16x16, 32x32.

Rubric:

1. Matrices are properly initialized.
2. Matrices are properly multiplied in GPU, and the result is validated in CPU.
3. GPU code is initialized correctly, and the device memory is deallocated.
4. Implement matrix multiplication using shared memory and tiling.
5. Report the average processing time and speedup for the different tile sizes.

## Results

8*8

Average timing:
1. CPU = 57029.5326 ms
2. GPU = 442.508232 ms
3. GPU tiling = 115.1819 ms

Speed up:
GPU - GPU TILING = 3.8418209

16*16

Average timing:
1. CPU = 57026.0131 ms
2. GPU = 778.107736 ms
3. GPU tiling = 67.4595319 ms

Speed up:
GPU - GPU TILING = 11.5344372

32*32

Average timing:
1. CPU = 56852.5791 ms
2. GPU = 1460.18385 ms
3. GPU tiling = 94.0656622 ms

Speed up:
GPU - GPU TILING = 15.5230274
