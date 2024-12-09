#pragma once


// 1. First ensure min/max macros don't interfere
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

// 2. Standard includes
#include <algorithm>
#include <cmath>

// 3. OpenCV includes
#include <opencv2/opencv.hpp>

// 4. Windows specific includes
#ifdef _MSC_VER
#define NOMINMAX  // Prevent Windows from defining min/max macros
#include <intrin.h>
#ifdef _WINDOWS
#include <Windows.h>
#endif
#elif defined(__GNUC__)
#include <x86intrin.h>
#endif

namespace match_utils {

// SIMD utility functions
inline int _mm_hsum_epi32(__m128i V) {
    __m128i T = _mm_add_epi32(V, _mm_srli_si128(V, 8));
    T = _mm_add_epi32(T, _mm_srli_si128(T, 4));
    return _mm_cvtsi128_si32(T);
}

inline int IM_Conv_SIMD(unsigned char* pCharKernel, unsigned char* pCharConv, int iLength) {
    const int iBlockSize = 16;
    const int Block = iLength / iBlockSize;
    __m128i SumV = _mm_setzero_si128();
    __m128i Zero = _mm_setzero_si128();

    for(int Y = 0; Y < Block * iBlockSize; Y += iBlockSize) {
        __m128i SrcK = _mm_loadu_si128((__m128i*)(pCharKernel + Y));
        __m128i SrcC = _mm_loadu_si128((__m128i*)(pCharConv + Y));
        __m128i SrcK_L = _mm_unpacklo_epi8(SrcK, Zero);
        __m128i SrcK_H = _mm_unpackhi_epi8(SrcK, Zero);
        __m128i SrcC_L = _mm_unpacklo_epi8(SrcC, Zero);
        __m128i SrcC_H = _mm_unpackhi_epi8(SrcC, Zero);
        __m128i SumT = _mm_add_epi32(_mm_madd_epi16(SrcK_L, SrcC_L),
                                    _mm_madd_epi16(SrcK_H, SrcC_H));
        SumV = _mm_add_epi32(SumV, SumT);
    }

    int Sum = _mm_hsum_epi32(SumV);
    for(int Y = Block * iBlockSize; Y < iLength; Y++) {
        Sum += pCharKernel[Y] * pCharConv[Y];
    }
    return Sum;
}

// Geometry utility functions
inline cv::Rect NormalizeRect(cv::Rect r) {
    if(r.width < 0) {
        r.x += r.width;
        r.width = -r.width;
    }
    if(r.height < 0) {
        r.y += r.height;
        r.height = -r.height;
    }
    return r;
}

inline cv::Point2f GetRectCenter(const cv::Rect& rect) {
    return cv::Point2f(rect.x + rect.width / 2.0f,
                      rect.y + rect.height / 2.0f);
}

// Math utility functions
inline double GetAngleDifference(double angle1, double angle2) {
    double diff = fabs(angle1 - angle2);
    return std::min(diff, 360.0 - diff);
}

inline void NormalizeAngle(double& angle) {
    while(angle > 180.0) angle -= 360.0;
    while(angle <= -180.0) angle += 360.0;
}

// Image processing utility functions
inline void FillBitmapInfo(BITMAPINFO* bmi, int width, int height, int bpp, int origin) {
    assert(bmi && width >= 0 && height >= 0 && (bpp == 8 || bpp == 24 || bpp == 32));

    BITMAPINFOHEADER* bmih = &(bmi->bmiHeader);
    memset(bmih, 0, sizeof(*bmih));
    bmih->biSize = sizeof(BITMAPINFOHEADER);
    bmih->biWidth = width;
    bmih->biHeight = origin ? abs(height) : -abs(height);
    bmih->biPlanes = 1;
    bmih->biBitCount = (unsigned short)bpp;
    bmih->biCompression = BI_RGB;

    if(bpp == 8) {
        RGBQUAD* palette = bmi->bmiColors;
        for(int i = 0; i < 256; i++) {
            palette[i].rgbBlue = palette[i].rgbGreen = palette[i].rgbRed = (BYTE)i;
            palette[i].rgbReserved = 0;
        }
    }
}

// Memory management utility functions
template<typename T>
inline void SafeDelete(T*& ptr) {
    if(ptr) {
        delete ptr;
        ptr = nullptr;
    }
}

template<typename T>
inline void SafeDeleteArray(T*& ptr) {
    if(ptr) {
        delete[] ptr;
        ptr = nullptr;
    }
}

// Debug utility functions
#ifdef _DEBUG
inline void DebugOutput(const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsprintf_s(buffer, format, args);
    va_end(args);
    OutputDebugStringA(buffer);
}
#else
inline void DebugOutput(const char*, ...) {}
#endif

} // namespace match_utils