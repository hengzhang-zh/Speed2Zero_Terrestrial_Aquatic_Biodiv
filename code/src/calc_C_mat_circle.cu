

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


extern "C" {
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define INDEX2D_YX(ncol, idy, idx) ((idy) * (ncol) + (idx))
#define INDEX3D_YXZ(nx, nz, idy, idx, idz) ((idy) * (nx) * (nz) + (idx) * (nz) + (idz))

#define PI 3.1415926
#define MAX_NUM_LYRS 24
#define MAX_SEARCH_DIST_RATIO 1.25


	__device__ float calc_C_r_elem(float dist_radius, float pix_dist, float pix_value, float pix_area, int decay_func) {
		float alpha, beta, s = 0.0;
		//decay function: exponential (exp)
		if (decay_func == 1) {
			alpha = 3.0 / dist_radius;
			beta = alpha;
			s = pix_value * pix_area * beta * cuda::std::expf(-1 * alpha * pix_dist);
		}
		//decay function: spherical (sph)
		else if (decay_func == 2) {
			float theta1 = dist_radius;
			float theta2 = (8.0 / 3 / theta1);
			if (pix_dist <= theta1) {
				s = pix_value * pix_area * theta2 * (1 - 1.5 * (pix_dist / theta1) + 0.5 * (pix_dist / theta1) * (pix_dist / theta1) * (pix_dist / theta1));
			}
		}
		//decay function: matern15
		else if (decay_func == 3) {
			alpha = 3.0 / dist_radius;
			beta = sqrtf(3.0) * alpha / 2;
			s = pix_value * pix_area * beta * (1 + sqrtf(3.0) * alpha * pix_dist) * cuda::std::expf(-1 * cuda::std::sqrtf(3.0) * alpha * pix_dist);
		}
		//decay function: matern25
		else if (decay_func == 4) {
			alpha = 3.0 / dist_radius;
			beta = 3.0 * sqrtf(5.0) * alpha / 8;
			s = pix_value * pix_area * beta * (1 + sqrtf(5.0) * alpha * pix_dist + 5.0 / 3 * alpha * alpha * pix_dist * pix_dist) *
				cuda::std::expf(-1 * cuda::std::sqrtf(5.0) * alpha * pix_dist);
		}
		//decay function: gaussian
		else if (decay_func == 5) {
			float alpha = 3 / dist_radius;
			float beta = 2 * sqrtf(alpha * alpha / PI / 3);
			s = pix_value * pix_area * beta * cuda::std::expf(-1 * alpha * alpha * pix_dist * pix_dist);
		}
		return s;
	}

	__global__ void calc_C_mat_circle(float* C_mat, float* ras_lyrs, float* A, float* pix_coords, float* dist_list, float* geotrans,
		int num_pix, int num_dist, int decay_func, int nrow, int ncol, int nlyr)
	{
		int i_pix = threadIdx.x + blockDim.x * blockIdx.x;
		if (i_pix >= num_pix)
			return;

		int2 point, i_point;
		point.x = int(0.5 + (pix_coords[2 * i_pix + 1] - geotrans[0]) / geotrans[1]);
		point.y = int(0.5 + (pix_coords[2 * i_pix] - geotrans[3]) / geotrans[5]);
		float pix_size = abs(geotrans[1]);
		float terra_effects[MAX_NUM_LYRS];
		for (int i = 0; i < MAX_NUM_LYRS; i++) {
			terra_effects[i] = 0.0;
		}

		//each pixel in the subset: search for circle distance buffer
		int2 i_start, i_end;
		for (int i_dist = 0; i_dist < num_dist; i_dist++) {
			float dist_radius = dist_list[i_dist];
			float pix_dist = 0.0;
			float pix_value, pix_area;
			int radius_pix = int(0.5 + dist_radius * MAX_SEARCH_DIST_RATIO / pix_size);
			i_start.y = MAX(point.y - radius_pix, 0);
			i_end.y = MIN(point.y + radius_pix + 1, nrow);
			i_start.x = MAX(point.x - radius_pix, 0);
			i_end.x = MIN(point.x + radius_pix + 1, ncol);

			//iteration within distance buffer (circle)
			for (i_point.y = i_start.y; i_point.y < i_end.y; i_point.y++) {
				for (i_point.x = i_start.x; i_point.x < i_end.x; i_point.x++) {
					pix_dist = pix_size * cuda::std::sqrtf((i_point.y - point.y) * (i_point.y - point.y) + (i_point.x - point.x) * (i_point.x - point.x));
					if (pix_dist > dist_radius * MAX_SEARCH_DIST_RATIO)
						continue;
					else {
						//calculation by raster layer
#pragma unroll
						for (int i_lyr = 0; i_lyr < nlyr; i_lyr++) {
							pix_value = ras_lyrs[INDEX3D_YXZ(ncol, nlyr, point.y, point.x, i_lyr)];
							pix_area = A[INDEX2D_YX(ncol, point.y, point.x)];
							terra_effects[i_lyr] += calc_C_r_elem(dist_radius, pix_dist, pix_value, pix_area, decay_func);
						}
					}
				}
			}
			//copy pixel level result back to C_mat 
#pragma unroll
			for (int i_lyr = 0; i_lyr < nlyr; i_lyr++) {
				C_mat[INDEX3D_YXZ(num_dist, nlyr, i_pix, i_dist, i_lyr)] = terra_effects[i_lyr];
				terra_effects[i_lyr] = 0.0;
			}
		}
	}

}