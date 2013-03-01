#pragma once

#include "colorstuff.h"

#ifdef __cplusplus
extern "C" {
#endif

// create a width x height window/surface
void sdls_init(unsigned int width, unsigned int height);

// cleanup
void sdls_cleanup(void);

// blit rectangles to window surface
void sdls_blitrectangle_rgba(unsigned int x, unsigned int y, unsigned int width, unsigned int height, const rgba * src);
void sdls_blitrectangle_grayscale(unsigned int x, unsigned int y, unsigned int width, unsigned int height, const grayscale * src);

// draw the image (flip buffers)
void sdls_draw(void);

// image loading functions
// returned buffers must be freed by the caller
rgba * sdls_loadimage_rgba(const char * file, size_t * width, size_t * height);
grayscale * sdls_loadimage_grayscale(const char * file, size_t * width, size_t * height);

#ifdef __cplusplus
}
#endif
