
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

#define MAX_LOOP 60000
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define DEFAULT_PIX_LEN 0.09
#define PI 3.1415926

__device__ bool updPosition(int2 &point, float &flowDC, int2 *dirvecs, float *DC_river, int *FD, int nrow, int ncol)
{
	bool bool_catch = false;
	int point_idx = point.y*ncol + point.x;
	if (DC_river[point_idx] > 1e-2) {
		//here this point is already in the major river channel
		bool_catch = true;
		flowDC = DC_river[point_idx];
	}
	else {
		int FD_value = FD[point_idx];
		int vec_idx = 0;
		if (FD_value == 0) {
			vec_idx = 0;
		}
		else {
			vec_idx = (int)log2f((float)FD_value) + 1;
		}
		//to update the point with current flow direction, then to fix out-of-boundary problem
		int2 point_upd;
		point_upd.x = point.x + dirvecs[vec_idx].x;
		point_upd.x = MAX(0, point_upd.x);
		point_upd.x = MIN(point_upd.x, ncol - 1);
		point_upd.y = point.y + dirvecs[vec_idx].y;
		point_upd.y = MAX(0, point_upd.y);
		point_upd.y = MIN(point_upd.y, nrow - 1);
		point = point_upd;
	}
	return bool_catch;
}

// catchment == 3: the outlet of catchment (should be defined in Python code); 
// catchmask == 2: in catchment; catchmask == 1: not in catchment; catchment == 0: to be decided
// In the catchdist array, the outflow points are indicated with a value -1
__global__ void intpFlowDischarge(float *DC, int *FD, float *DC_river, int nrow, int ncol)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	if (x >= ncol || y >= nrow)
		return;
	//set up flow direction vectors & distance vectors
	int2 dirvecs[9];
	//case in Thailand 
	dirvecs[0].y = 0; dirvecs[0].x = 0; dirvecs[1].y = 0; dirvecs[1].x = 1; dirvecs[2].y = 1; dirvecs[2].x = 1; dirvecs[3].y = 1; dirvecs[3].x = 0;
	dirvecs[4].y = 1; dirvecs[4].x = -1; dirvecs[5].y = 0; dirvecs[5].x = -1; dirvecs[6].y = -1; dirvecs[6].x = -1; dirvecs[7].y = -1; dirvecs[7].x = 0;
	dirvecs[8].y = -1; dirvecs[8].x = 1;
	//float distvalues[9] = { 0, 1, 1.41421356237309504880, 1, 1.41421356237309504880, 1, 1.41421356237309504880, 1, 1.41421356237309504880 };

	//set up the intial conditions
	int2 point, point_prev;
	point.x = x;	point.y = y;
	point_prev = point;
	int idx = y*ncol + x;
	float flowDC = 0.0;
	bool bool_catch = false;


	for (int i = 0; i < MAX_LOOP; i++) {
		//to track water flow of the point and to check whether it is in the main river channel
		bool_catch = updPosition(point, flowDC, dirvecs, DC_river, FD, nrow, ncol);
		if (bool_catch) {
			DC[idx] = flowDC;
			break;
		}
		//if the flow point is repeating (reached a local minimum)
		else if (point.x == point_prev.x&&point.y == point_prev.y) {
			break;
		}
		point_prev = point;
	}
	if (!bool_catch) {
		DC[idx] = 0;
	}
}
