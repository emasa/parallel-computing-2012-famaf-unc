#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "SDL/SDL.h"
#include "SDL/SDL_video.h"
#include "SDL/SDL_image.h"

#include "colorstuff.h"
#include "sdlstuff.h"

#define PIXEL_DEPTH 32


void sdls_init(unsigned int width, unsigned int height) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Unable to init SDL: %s\n", SDL_GetError());
        exit(1);
    }
    atexit(sdls_cleanup);

    SDL_Surface * window_surface = SDL_SetVideoMode(width, height, PIXEL_DEPTH, SDL_SWSURFACE | SDL_DOUBLEBUF);
    if (window_surface == NULL) {
        fprintf(stderr, "Couldn't create window: %s\n", SDL_GetError());
        exit(1);
    }
}


void sdls_cleanup(void) {
    SDL_Quit();
}


static void sdls_setpixel(SDL_Surface * surface, unsigned int x, unsigned int y, rgba color) {
    Uint32 value = SDL_MapRGBA(surface->format, color.r, color.g, color.b, color.a);
    Uint32 * pixel = (Uint32*) surface->pixels + y * (surface->pitch / surface->format->BytesPerPixel) + x;
    *pixel = value;
}


static void sdls_setpixel(SDL_Surface * surface, unsigned int x, unsigned int y, grayscale color) {
    Uint32 value = SDL_MapRGBA(surface->format, color, color, color, 255);
    Uint32 * pixel = (Uint32*) surface->pixels + y * (surface->pitch / surface->format->BytesPerPixel) + x;
    *pixel = value;
}


template<typename T>
static void sdls_blitrectangle(unsigned int x, unsigned int y, unsigned int width, unsigned int height, const T * src) {
    SDL_Surface * window_surface = SDL_GetVideoSurface();

    if (SDL_MUSTLOCK(window_surface)) {
        SDL_LockSurface(window_surface);
    }

    // FIXME: poor man's blit
    const T * cursor = src;
    for (unsigned int row = y; row < y + height; ++row) {
        for (unsigned int col = x; col < x + width; ++col) {
            sdls_setpixel(window_surface, col, row, *cursor);
            ++cursor;
        }
    }

    if (SDL_MUSTLOCK(window_surface)) {
        SDL_UnlockSurface(window_surface);
    }

}


void sdls_blitrectangle_rgba(unsigned int x, unsigned int y, unsigned int width, unsigned int height, const rgba * src) {
    sdls_blitrectangle(x, y, width, height, src);
}


void sdls_blitrectangle_grayscale(unsigned int x, unsigned int y, unsigned int width, unsigned int height, const grayscale * src) {
    sdls_blitrectangle(x, y, width, height, src);
}


void sdls_draw(void) {
    SDL_Surface * window_surface = SDL_GetVideoSurface();
    SDL_Flip(window_surface);
}


static grayscale rgba_to_gray(rgba in) {
    return (grayscale) rintf(0.2989f * in.r + 0.5870f * in.g + 0.1140f * in.b);
}


static Uint32 get_pixel(SDL_Surface *surface, int x, int y) {
    int bpp = surface->format->BytesPerPixel;
    /* Here p is the address to the pixel we want to retrieve */
    Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * bpp;

    switch(bpp) {
    case 1:
        return *p;
        break;

    case 2:
        return *(Uint16 *)p;
        break;

    case 3:
        if(SDL_BYTEORDER == SDL_BIG_ENDIAN) {
            return p[0] << 16 | p[1] << 8 | p[2];
        } else {
            return p[0] | p[1] << 8 | p[2] << 16;
        }
        break;

    case 4:
        return *(Uint32 *)p;
        break;

    default:
        return 0;       /* shouldn't happen, but avoids warnings */
    }
}


rgba * sdls_loadimage_rgba(const char * file, size_t * width, size_t * height) {
    SDL_Surface * image_surface;
    image_surface = IMG_Load(file);
    if (image_surface == 0) {
        fprintf(stderr, "Couldn't load image %s: %s\n", file, IMG_GetError());
        return 0;
    }

    *width = image_surface->w;
    *height = image_surface->h;
    SDL_PixelFormat * format = image_surface->format;

    rgba * result = (rgba *) malloc(image_surface->w * image_surface->h * sizeof(rgba));
    for (int y = 0; y < image_surface->h; ++y) {
        for (int x = 0; x < image_surface->w; ++x) {
            rgba color;
            Uint32 pixel = get_pixel(image_surface, x, y);
            SDL_GetRGBA(pixel, format, &color.r, &color.g, &color.b, &color.a);
            result[x + y * image_surface->w] = color;
        }
    }
    SDL_FreeSurface(image_surface);

    return result;
}


grayscale * sdls_loadimage_grayscale(const char * file, size_t * width, size_t * height) {
    SDL_Surface * image_surface;
    image_surface = IMG_Load(file);
    if (image_surface == 0) {
        fprintf(stderr, "Couldn't load image %s: %s\n", file, IMG_GetError());
        return 0;
    }

    *width = image_surface->w;
    *height = image_surface->h;
    SDL_PixelFormat * format = image_surface->format;

    grayscale * result = (grayscale *) malloc(image_surface->w * image_surface->h * sizeof(grayscale));
    for (int y = 0; y < image_surface->h; ++y) {
        for (int x = 0; x < image_surface->w; ++x) {
            rgba color;
            Uint32 pixel = get_pixel(image_surface, x, y);
            SDL_GetRGBA(pixel, format, &color.r, &color.g, &color.b, &color.a);
            result[x + y * image_surface->w] = rgba_to_gray(color);
        }
    }
    SDL_FreeSurface(image_surface);

    return result;
}
