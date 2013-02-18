#include <stddef.h>
#include <stdio.h>
#include <x86intrin.h>  // soporte para intrisics

#define IX(i,j) ((i)+(n+2)*(j))

#define SHIFT_LEFT(_mm_value) _mm_shuffle_ps((_mm_value), (_mm_value), _MM_SHUFFLE(0, 3, 2, 1))  
#define SHIFT_RIGHT(_mm_value) _mm_shuffle_ps((_mm_value), (_mm_value), _MM_SHUFFLE(2, 1, 0, 3))  

int main(){
    
    //u[IX(i + 0, j)] -= 0.5f * n * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
    //u[IX(i + 1, j)] -= 0.5f * n * (p[IX(i + 2, j)] - p[IX(i - 0, j)]);            
    
    size_t n = 4, dummy = 0;
    float u[] = {2., 2., 2., 2.};
    
    __m128 n_div_2 = _mm_set1_ps(n / 2.);
    __m128 p_center = { 1., 2., 3., 5. };
    dummy++;
    __m128 p_sub = _mm_sub_ps(SHIFT_LEFT(p_center), SHIFT_RIGHT(p_center));
    __m128 p_sub_mul_n_div_2 = _mm_mul_ps(p_sub, n_div_2); 
    dummy++;
    __m128 u_center = _mm_loadu_ps((float*) &u[0]);
    __m128 u_sub_p  = _mm_sub_ps(u_center, p_sub_mul_n_div_2);
    dummy++;
    //__m128 u_res = _mm_blend_ps(u_center, u_sub_p, 6); //0110
    __m128 u_res_aux = _mm_shuffle_ps(u_sub_p, u_center, _MM_SHUFFLE(3, 0, 1, 2));
    __m128 u_res = _mm_shuffle_ps(u_res_aux, u_res_aux, _MM_SHUFFLE(3, 0, 1, 2));
    dummy++;
    _mm_storeu_ps((float*) &u[0], u_res);
    dummy++;    
    printf("%f %f %f %f\n", u[0], u[1], u[2], u[3]);
     
    return 0;
}
