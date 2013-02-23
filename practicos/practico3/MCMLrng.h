#pragma once

//Generate a random number [0,1)
float rand_MWC_co(unsigned long long * x, unsigned int * a);

//Generate a random number (0,1]
float rand_MWC_oc(unsigned long long * x, unsigned int * a);

int init_RNG(unsigned long long *x, unsigned int *a,
             const unsigned int n_rng, const char *safeprimes_file, unsigned long long xinit);
