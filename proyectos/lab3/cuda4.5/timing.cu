//
// timing.c
//

#include <sys/time.h>

#include "timing.h"

double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, 0);

  return (double) tv.tv_sec + 1e-6 * tv.tv_usec;
}
