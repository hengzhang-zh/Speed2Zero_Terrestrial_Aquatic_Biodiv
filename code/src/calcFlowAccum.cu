
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cmath>


#define MAX_LOOP 60000
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

typedef unsigned char uchar;

__device__ void updPosition(int2& point, const int2* dirvecs, const uchar* FD, const int nrow, const int ncol)
{
	int point_idx = point.y * ncol + point.x;
	uchar FD_value = FD[point_idx];
	uchar vec_idx = 0;
	if (FD_value == 0) {
		vec_idx = 0;
	}
	else {
		vec_idx = (int)log2f((float)FD_value) + 1;
	}
	//update the point with current flow direction, then to fix out-of-boundary problem
	int2 point_upd;
	point_upd.x = point.x + dirvecs[vec_idx].x;
	point_upd.x = MAX(0, point_upd.x);
	point_upd.x = MIN(point_upd.x, ncol - 1);
	point_upd.y = point.y + dirvecs[vec_idx].y;
	point_upd.y = MAX(0, point_upd.y);
	point_upd.y = MIN(point_upd.y, nrow - 1);
	point = point_upd;
}


__global__ void calcFlowAccum(int* FA, const uchar* FD, const int nrow, const int ncol)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= ncol || y >= nrow)
		return;
	//set up flow direction vectors & distance vectors
	int2 dirvecs[9];

	//case in ArcGIS & HydroSHEDS v1.0
	dirvecs[0].y = 0; dirvecs[0].x = 0; dirvecs[1].y = 0; dirvecs[1].x = 1; dirvecs[2].y = 1; dirvecs[2].x = 1; dirvecs[3].y = 1; dirvecs[3].x = 0;
	dirvecs[4].y = 1; dirvecs[4].x = -1; dirvecs[5].y = 0; dirvecs[5].x = -1; dirvecs[6].y = -1; dirvecs[6].x = -1; dirvecs[7].y = -1; dirvecs[7].x = 0;
	dirvecs[8].y = -1; dirvecs[8].x = 1;

	//set up the intial conditions
	int2 point, point_prev;
	point.x = x;	point.y = y;
	point_prev = point;


	//to track water flow of the point and to check whether it is in the catchment
	for (int i = 0; i < MAX_LOOP; i++) {
		updPosition(point, dirvecs, FD, nrow, ncol);
		//if the flow point is repeating (reached a local minimum)
		if (point.x == point_prev.x && point.y == point_prev.y)
			break;
		//update Flow Accumulation Map
		atomicAdd(&FA[point.y * ncol + point.x], 1);

		//check whether the point reaches boundary
		if (point.x == 0 || point.x == (ncol - 1) || point.y == 0 || point.y == (nrow - 1))
			break;
		point_prev = point;
	}

}
