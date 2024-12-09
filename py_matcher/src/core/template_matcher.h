#pragma once
#include <algorithm>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "match_types.h"
#include "match_utils.h"  // 确保这个包含存在

class TemplateMatcher {
public:
    TemplateMatcher();
    ~TemplateMatcher();

    // Main functions from MFC version
    bool Match();
    void LearnPattern();
    void FilterWithScore(std::vector<s_MatchParameter>* vec, double dScore);
    void FilterWithRotatedRect(std::vector<s_MatchParameter>* vec, int iMethod = CV_TM_CCOEFF_NORMED, double dMaxOverLap = 0);

    // Core matching functions
    void MatchTemplate(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, bool bUseSIMD);
    void CCOEFF_Denominator(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer);
    bool SubPixEsimation(std::vector<s_MatchParameter>* vec, double* dNewX, double* dNewY, double* dNewAngle, double dAngleStep, int iMaxScoreIndex);

    // Utility functions
    int GetTopLayer(cv::Mat* matTempl, int iMinDstLength);
    void GetRotatedROI(cv::Mat& matSrc, cv::Size size, cv::Point2f ptLT, double dAngle, cv::Mat& matROI);
    cv::Size GetBestRotationSize(cv::Size sizeSrc, cv::Size sizeDst, double dRAngle);
    cv::Point2f ptRotatePt2f(cv::Point2f ptInput, cv::Point2f ptOrg, double dAngle);
    cv::Point GetNextMaxLoc(cv::Mat& matResult, cv::Point ptMaxLoc, cv::Size sizeTemplate, double& dMaxValue, double dMaxOverlap);

    // Drawing functions
    void DrawDashLine(cv::Mat& matDraw, cv::Point ptStart, cv::Point ptEnd, cv::Scalar color1 = cv::Scalar(0,0,255), cv::Scalar color2 = cv::Scalar::all(255));
    void DrawMarkCross(cv::Mat& matDraw, int iX, int iY, int iLength, cv::Scalar color, int iThickness);

    // Parameter settings
    void SetMaxPositions(int value) { m_iMaxPos = value; }
    void SetMaxOverlap(double value) { m_dMaxOverlap = value; }
    void SetScore(double value) { m_dScore = value; }
    void SetToleranceAngle(double value) { m_dToleranceAngle = value; }
    void SetMinReduceArea(int value) { m_iMinReduceArea = value; }
    void SetDebugMode(bool value) { m_bDebugMode = value; }
    void SetToleranceRangeMode(bool value) { m_bToleranceRange = value; }
    void SetToleranceRanges(double t1, double t2, double t3, double t4) {
        m_dTolerance1 = t1;
        m_dTolerance2 = t2;
        m_dTolerance3 = t3;
        m_dTolerance4 = t4;
    }
    void SetStopLayer1(bool value) { m_bStopLayer1 = value; }
    void SetUseSIMD(bool value) { m_bUseSIMD = value; }
    void SetUseSubPixel(bool value) { m_bUseSubPixel = value; }

    // Results access
    const std::vector<s_SingleTargetMatch>& GetResults() const { return m_vecSingleTargetData; }
    
    // Image setting
    void SetSourceImage(const cv::Mat& img) { m_matSrc = img.clone(); }
    void SetTemplateImage(const cv::Mat& img) { m_matDst = img.clone(); }

private:
    // Member variables from MFC version
    cv::Mat m_matSrc;
    cv::Mat m_matDst;
    s_TemplData m_TemplData;
    std::vector<s_SingleTargetMatch> m_vecSingleTargetData;

    // Parameters
    int m_iMaxPos;
    double m_dMaxOverlap;
    double m_dScore;
    double m_dToleranceAngle;
    int m_iMinReduceArea;
    bool m_bDebugMode;
    bool m_bToleranceRange;
    double m_dTolerance1;
    double m_dTolerance2;
    double m_dTolerance3;
    double m_dTolerance4;
    bool m_bStopLayer1;
    bool m_bUseSIMD;
    bool m_bUseSubPixel;

    // Block maximum calculation structure
    struct s_BlockMax {
        struct Block {
            cv::Rect rect;
            double dMax;
            cv::Point ptMaxLoc;
        };
        std::vector<Block> vecBlock;
        cv::Mat matSrc;

        s_BlockMax(cv::Mat& src, cv::Size sizeTemplate);
        void UpdateMax(cv::Rect rectIgnore);
        void GetMaxValueLoc(double& dMax, cv::Point& ptMaxLoc);
    };

    cv::Point GetNextMaxLoc(cv::Mat& matResult, cv::Point ptMaxLoc, cv::Size sizeTemplate, 
                         double& dMaxValue, double dMaxOverlap, s_BlockMax& blockMax);
};