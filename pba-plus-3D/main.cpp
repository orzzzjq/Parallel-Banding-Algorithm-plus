/*
Author: Cao Thanh Tung, Zheng Jiaqi
Date: 21/01/2010, 25/08/2019

File Name: main.cpp

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "pba\pba3D.h"

// Input parameters
int fboSize     = 1024;
int nVertices   = 100;

int phase1Band  = 1;
const int phase2Band = 1; 
int phase3Band	= 2;

#define TOID(x, y, z, w)    ((z) * (w) * (w) + (y) * (w) + (x))

typedef struct {
    double totalDistError, maxDistError; 
    int errorCount; 
} ErrorStatistics; 

// Global Vars
int *inputPoints, *inputVoronoi, *outputVoronoi; 
ErrorStatistics pba; 

// Random Point Generator
// Random number generator, obtained from http://oldmill.uchicago.edu/~wilder/Code/random/
unsigned long z, w, jsr, jcong; // Seeds
void randinit(unsigned long x_) 
{ z =x_; w = x_; jsr = x_; jcong = x_; }
unsigned long znew() 
{ return (z = 36969 * (z & 0xfffful) + (z >> 16)); }
unsigned long wnew() 
{ return (w = 18000 * (w & 0xfffful) + (w >> 16)); }
unsigned long MWC()  
{ return ((znew() << 16) + wnew()); }
unsigned long SHR3()
{ jsr ^= (jsr << 17); jsr ^= (jsr >> 13); return (jsr ^= (jsr << 5)); }
unsigned long CONG() 
{ return (jcong = 69069 * jcong + 1234567); }
unsigned long rand_int()         // [0,2^32-1]
{ return ((MWC() ^ CONG()) + SHR3()); }
double random()     // [0,1)
{ return ((double) rand_int() / (double(ULONG_MAX)+1)); }

// Generate input points
void generateRandomPoints(int texSize, int nPoints)
{	
    int tx, ty, tz, id; 

    randinit(0);

    for (int i = 0; i < texSize * texSize * texSize; i++)
        inputVoronoi[i] = MARKER; 

	for (int i = 0; i < nPoints; i++)
	{
        do { 
            tx = int(random() * texSize); 
            ty = int(random() * texSize); 
            tz = int(random() * texSize); 
            id = TOID(tx, ty, tz, texSize); 
		} while (inputVoronoi[id] != MARKER); 

        inputVoronoi[id] = ENCODE(tx, ty, tz, 0, 0); 
        inputPoints[i] = ENCODE(tx, ty, tz, 0, 0);
    }
}

// Deinitialization
void deinitialization()
{
    pba3DDeinitialization(); 

    free(inputPoints); 
    free(inputVoronoi); 
    free(outputVoronoi); 
}

// Initialization
void initialization()
{
    pba3DInitialization(fboSize); 

    inputPoints     = (int *) malloc(nVertices * sizeof(int));
    inputVoronoi    = (int *) malloc(fboSize * fboSize * fboSize * sizeof(int));
    outputVoronoi   = (int *) malloc(fboSize * fboSize * fboSize * sizeof(int));
}

// Verify the output Voronoi Diagram
void compareResult(ErrorStatistics *e) 
{
    e->totalDistError = 0.0; 
    e->maxDistError = 0.0; 
    e->errorCount = 0; 

    int dx, dy, dz, nx, ny, nz; 
    double dist, myDist, correctDist, error;

    for (int k = 0; k < fboSize; k++) {
        for (int i = 0; i < fboSize; i++) {
            for (int j = 0; j < fboSize; j++) {
                int id = TOID(i, j, k, fboSize); 
                DECODE(outputVoronoi[id], nx, ny, nz); 

                dx = nx - i; dy = ny - j; dz = nz - k; 
                correctDist = myDist = dx * dx + dy * dy + dz * dz; 

                for (int t = 0; t < nVertices; t++) {
                    DECODE(inputPoints[t], nx, ny, nz); 
                    dx = nx - i; dy = ny - j; dz = nz - k; 
                    dist = dx * dx + dy * dy + dz * dz; 

                    if (dist < correctDist)
                        correctDist = dist; 
                }

                if (correctDist != myDist) {
                    error = fabs(sqrt(myDist) - sqrt(correctDist)); 

                    e->errorCount++; 
                    e->totalDistError += error; 

                    if (error > e->maxDistError)
                        e->maxDistError = error; 
                }
            }
        }
    }
}

void printStatistics(ErrorStatistics *e)
{
    double avgDistError = e->totalDistError / e->errorCount; 

    if (e->errorCount == 0)
        avgDistError = 0.0; 

    printf("* Error count           : %i -> %.3f%\n", e->errorCount, 
        (double(e->errorCount) / nVertices) * 100.0);
    printf("* Max distance error    : %.5f\n", e->maxDistError);
    printf("* Average distance error: %.5f\n", avgDistError);
}

// Run the tests
void runTests()
{
    generateRandomPoints(fboSize, nVertices); 

	pba3DVoronoiDiagram(inputVoronoi, outputVoronoi, phase1Band, phase2Band, phase3Band); 

    printf("Verifying the result...\n"); 
    compareResult(&pba);

    printf("-----------------\n");
    printf("Texture: %dx%dx%d\n", fboSize, fboSize, fboSize);
    printf("Points: %d\n", nVertices);
    printf("-----------------\n");

    printStatistics(&pba); 
}

int main(int argc, char **argv)
{
    initialization();

    runTests();

    deinitialization();

	return 0;
}