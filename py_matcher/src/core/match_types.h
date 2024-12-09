#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/imgproc/imgproc_c.h>

// OpenCV 3/4 compatibility
#if CV_MAJOR_VERSION >= 4
#define CV_TM_SQDIFF        cv::TM_SQDIFF
#define CV_TM_SQDIFF_NORMED cv::TM_SQDIFF_NORMED
#define CV_TM_CCORR        cv::TM_CCORR
#define CV_TM_CCORR_NORMED cv::TM_CCORR_NORMED
#define CV_TM_CCOEFF       cv::TM_CCOEFF
#define CV_TM_CCOEFF_NORMED cv::TM_CCOEFF_NORMED
#endif

// 使用std::max 而不是 max
#include <algorithm>
using std::max;

// Constants from original MFC code
#define VISION_TOLERANCE 0.0000001
#define D2R (CV_PI / 180.0)
#define R2D (180.0 / CV_PI)
#define MATCH_CANDIDATE_NUM 5
#define MAX_SCALE_TIMES 10
#define MIN_SCALE_TIMES 0
#define SCALE_RATIO 1.25

// Match result structure
struct s_MatchParameter 
{
    cv::Point2d pt;
    double dMatchScore;
    double dMatchAngle;
    cv::Rect rectRoi;
    double dAngleStart;
    double dAngleEnd;
    cv::RotatedRect rectR;
    cv::Rect rectBounding;
    bool bDelete;
    double vecResult[3][3];  // for subpixel
    int iMaxScoreIndex;      // for subpixel
    bool bPosOnBorder;
    cv::Point2d ptSubPixel;
    double dNewAngle;

    s_MatchParameter(cv::Point2f ptMinMax, double dScore, double dAngle)
        : pt(ptMinMax)
        , dMatchScore(dScore)
        , dMatchAngle(dAngle)
        , bDelete(false)
        , dNewAngle(0.0)
        , bPosOnBorder(false)
    {}

    s_MatchParameter()
        : dMatchScore(0)
        , dMatchAngle(0)
    {}
};

struct s_SingleTargetMatch 
{
    cv::Point2d ptLT, ptRT, ptRB, ptLB, ptCenter;
    double dMatchedAngle;
    double dMatchScore;
};

struct s_TemplData 
{
    std::vector<cv::Mat> vecPyramid;
    std::vector<cv::Scalar> vecTemplMean;
    std::vector<double> vecTemplNorm;
    std::vector<double> vecInvArea;
    std::vector<bool> vecResultEqual1;
    bool bIsPatternLearned;
    int iBorderColor;

    void clear() 
    {
        vecPyramid.clear();
        vecTemplNorm.clear();
        vecInvArea.clear();
        vecTemplMean.clear();
        vecResultEqual1.clear();
    }

    void resize(int iSize) 
    {
        vecTemplMean.resize(iSize);
        vecTemplNorm.resize(iSize, 0);
        vecInvArea.resize(iSize, 1);
        vecResultEqual1.resize(iSize, false);
    }

    s_TemplData()
        : bIsPatternLearned(false) 
    {}
};

// Compare functions from original code
inline bool compareScoreBig2Small(const s_MatchParameter& lhs, const s_MatchParameter& rhs)
{
    return lhs.dMatchScore > rhs.dMatchScore;
}

inline bool comparePtWithAngle(const std::pair<cv::Point2f, double> lhs, const std::pair<cv::Point2f, double> rhs)
{
    return lhs.second < rhs.second;
}

inline bool compareMatchResultByPos(const s_SingleTargetMatch& lhs, const s_SingleTargetMatch& rhs)
{
    double dTol = 2;
    if (fabs(lhs.ptCenter.y - rhs.ptCenter.y) <= dTol)
        return lhs.ptCenter.x < rhs.ptCenter.x;
    else
        return lhs.ptCenter.y < rhs.ptCenter.y;
}

inline bool compareMatchResultByScore(const s_SingleTargetMatch& lhs, const s_SingleTargetMatch& rhs)
{
    return lhs.dMatchScore > rhs.dMatchScore;
}

inline bool compareMatchResultByPosX(const s_SingleTargetMatch& lhs, const s_SingleTargetMatch& rhs)
{
    return lhs.ptCenter.x < rhs.ptCenter.x;
}