/*
Author: Cao Thanh Tung, Zheng Jiaqi
Date: 21/01/2010, 25/08/2019

File Name: pba3DHost.cu

===============================================================================

Copyright (c) 2019, School of Computing, National University of Singapore. 
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/pba.html

If you use PBA and you like it or have comments on its usefulness etc., we 
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

===============================================================================

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <device_functions.h>
#include <helper_timer.h>

#include "pba3D.h"

// Parameters for CUDA kernel executions
#define BLOCKX      32
#define BLOCKY      4
#define BLOCKSIZE 	32

// Global Variables
int **pbaTextures;

size_t pbaMemSize; 
int pbaCurrentBuffer; 
int pbaTexSize;

texture<int> pbaTexColor; 
texture<int> pbaTexLinks; 

// Kernels
#include "pba3DKernel.h"

// Initialize necessary memory for 3D Voronoi Diagram computation
// - textureSize: The size of the Discrete Voronoi Diagram (width = height)
void pba3DInitialization(int fboSize)
{
   	pbaTexSize = fboSize; 

	pbaTextures = (int **) malloc(2 * sizeof(int *)); 

	pbaMemSize = pbaTexSize * pbaTexSize * pbaTexSize * sizeof(int); 

	cudaMalloc((void **) &pbaTextures[0], pbaMemSize);
	cudaMalloc((void **) &pbaTextures[1], pbaMemSize);
}

// Deallocate all allocated memory
void pba3DDeinitialization()
{
	cudaFree(pbaTextures[0]); 
	cudaFree(pbaTextures[1]); 

	free(pbaTextures); 
}

// Copy input to GPU 
void pba3DInitializeInput(int *input)
{
    cudaMemcpy(pbaTextures[0], input, pbaMemSize, cudaMemcpyHostToDevice); 

	pbaCurrentBuffer = 0;
}

void pba3DColorZAxis(int m1) 
{
   	dim3 block = dim3(BLOCKX, BLOCKY); 
    dim3 grid = dim3(pbaTexSize / block.x, pbaTexSize / block.y); 

    kernelFloodZ<<< grid, block >>>(pbaTextures[pbaCurrentBuffer], pbaTextures[1 - pbaCurrentBuffer], pbaTexSize); 
    pbaCurrentBuffer = 1 - pbaCurrentBuffer; 
}

void pba3DComputeProximatePointsYAxis(int m2) 
{
	dim3 block = dim3(BLOCKX, BLOCKY); 
    dim3 grid = dim3(pbaTexSize / block.x, pbaTexSize / block.y); 

    kernelMaurerAxis<<< grid, block >>>(pbaTextures[pbaCurrentBuffer], pbaTextures[1 - pbaCurrentBuffer], pbaTexSize); 
}

// Phase 3 of PBA. m3 must divides texture size
// This method color along the Y axis
void pba3DColorYAxis(int m3) 
{
	dim3 block = dim3(BLOCKSIZE, m3); 
    dim3 grid = dim3(pbaTexSize / block.x, pbaTexSize); 

    kernelColorAxis<<< grid, block >>>(pbaTextures[1 - pbaCurrentBuffer], pbaTextures[pbaCurrentBuffer], pbaTexSize); 
}

void pba3DCompute(int m1, int m2, int m3)
{
	pba3DColorZAxis(m1); 

	pba3DComputeProximatePointsYAxis(m2);

	pba3DColorYAxis(m3); 

	pba3DComputeProximatePointsYAxis(m2);

	pba3DColorYAxis(m3); 
}

// Compute 3D Voronoi diagram
// Input: a 3D texture. Each pixel is an integer encoding 3 coordinates. 
// 		For each site at (x, y, z), the pixel at coordinate (x, y, z) should contain 
// 		the encoded coordinate (x, y, z). Pixels that are not sites should contain 
// 		the integer MARKER. Use ENCODE (and DECODE) macro to encode (and decode).
// See our website for the effect of the three parameters: 
// 		phase1Band, phase2Band, phase3Band
// Parameters must divide textureSize
void pba3DVoronoiDiagram(int *input, int *output, 
                         int phase1Band, int phase2Band, int phase3Band) 
{
	// Initialization
	pba3DInitializeInput(input); 

    // Compute the 3D Voronoi Diagram
    pba3DCompute(phase1Band, phase2Band, phase3Band); 

    // Copy back the result
    cudaMemcpy(output, pbaTextures[pbaCurrentBuffer], pbaMemSize, cudaMemcpyDeviceToHost); 
}

