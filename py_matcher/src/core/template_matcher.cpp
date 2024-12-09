#include "template_matcher.h"
#include <algorithm>
using std::min;  // 如果你想直接使用 min

// 构造函数
TemplateMatcher::TemplateMatcher() :
    m_iMaxPos(7),
    m_dMaxOverlap(0),
    m_dScore(0.5),
    m_dToleranceAngle(0),
    m_iMinReduceArea(256),
    m_bDebugMode(false),
    m_dTolerance1(40),
    m_dTolerance2(60),
    m_dTolerance3(-110),
    m_dTolerance4(-100),
    m_bStopLayer1(false),
    m_bToleranceRange(false),
    m_bUseSIMD(true),
    m_bUseSubPixel(false)
{
}

TemplateMatcher::~TemplateMatcher()
{
}

bool TemplateMatcher::Match()
{
    if (m_matSrc.empty() || m_matDst.empty())
        return false;
        
    if ((m_matDst.cols < m_matSrc.cols && m_matDst.rows > m_matSrc.rows) || 
        (m_matDst.cols > m_matSrc.cols && m_matDst.rows < m_matSrc.rows))
        return false;
        
    if (m_matDst.size().area() > m_matSrc.size().area())
        return false;
        
    if (!m_TemplData.bIsPatternLearned)
        return false;

    // 决定金字塔层数
    int iTopLayer = GetTopLayer(&m_matDst, (int)sqrt((double)m_iMinReduceArea));

    // 建立金字塔
    std::vector<cv::Mat> vecMatSrcPyr;
    buildPyramid(m_matSrc, vecMatSrcPyr, iTopLayer);

    s_TemplData* pTemplData = &m_TemplData;

    // 第一阶段以最顶层找出大致角度与ROI
    double dAngleStep = atan(2.0 / max(pTemplData->vecPyramid[iTopLayer].cols, 
                                     pTemplData->vecPyramid[iTopLayer].rows)) * R2D;

    std::vector<double> vecAngles;

    // 生成搜索角度
    if (m_bToleranceRange)
    {
        if (m_dTolerance1 >= m_dTolerance2 || m_dTolerance3 >= m_dTolerance4)
            return false;
            
        for(double dAngle = m_dTolerance1; dAngle < m_dTolerance2 + dAngleStep; dAngle += dAngleStep)
            vecAngles.push_back(dAngle);
            
        for(double dAngle = m_dTolerance3; dAngle < m_dTolerance4 + dAngleStep; dAngle += dAngleStep)
            vecAngles.push_back(dAngle);
    }
    else
    {
        if (m_dToleranceAngle < VISION_TOLERANCE)
            vecAngles.push_back(0.0);
        else 
        {
            for(double dAngle = 0; dAngle < m_dToleranceAngle + dAngleStep; dAngle += dAngleStep)
                vecAngles.push_back(dAngle);
            for(double dAngle = -dAngleStep; dAngle > -m_dToleranceAngle - dAngleStep; dAngle -= dAngleStep)
                vecAngles.push_back(dAngle);
        }
    }

    // 匹配处理
    int iTopSrcW = vecMatSrcPyr[iTopLayer].cols;
    int iTopSrcH = vecMatSrcPyr[iTopLayer].rows;
    cv::Point2f ptCenter((iTopSrcW - 1) / 2.0f, (iTopSrcH - 1) / 2.0f);

    std::vector<s_MatchParameter> vecMatchParameter;
    std::vector<double> vecLayerScore(iTopLayer + 1, m_dScore);
    
    for(int iLayer = 1; iLayer <= iTopLayer; iLayer++)
        vecLayerScore[iLayer] = vecLayerScore[iLayer - 1] * 0.9;

    cv::Size sizePat = pTemplData->vecPyramid[iTopLayer].size();
    bool bCalMaxByBlock = (vecMatSrcPyr[iTopLayer].size().area() / sizePat.area() > 500) && m_iMaxPos > 10;

    // Top layer matching
    for(double dAngle : vecAngles)
    {
        cv::Mat matRotatedSrc;
        cv::Mat matR = cv::getRotationMatrix2D(ptCenter, dAngle, 1);
        cv::Mat matResult;

        cv::Size sizeBest = GetBestRotationSize(vecMatSrcPyr[iTopLayer].size(), 
                                              pTemplData->vecPyramid[iTopLayer].size(), dAngle);

        float fTranslationX = (sizeBest.width - 1) / 2.0f - ptCenter.x;
        float fTranslationY = (sizeBest.height - 1) / 2.0f - ptCenter.y;
        matR.at<double>(0, 2) += fTranslationX;
        matR.at<double>(1, 2) += fTranslationY;

        cv::warpAffine(vecMatSrcPyr[iTopLayer], matRotatedSrc, matR, sizeBest,
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, 
                      cv::Scalar(pTemplData->iBorderColor));

        MatchTemplate(matRotatedSrc, pTemplData, matResult, iTopLayer, false);

        if(bCalMaxByBlock)
        {
            s_BlockMax blockMax(matResult, pTemplData->vecPyramid[iTopLayer].size());
            cv::Point ptMaxLoc;
            double dMaxVal;
            blockMax.GetMaxValueLoc(dMaxVal, ptMaxLoc);

            if(dMaxVal < vecLayerScore[iTopLayer])
                continue;

            vecMatchParameter.push_back(s_MatchParameter(
                cv::Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY),
                dMaxVal, dAngle));

            for(int j = 0; j < m_iMaxPos + MATCH_CANDIDATE_NUM - 1; j++)
            {
                ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, 
                                       pTemplData->vecPyramid[iTopLayer].size(),
                                       dMaxVal, m_dMaxOverlap, blockMax);
                                       
                if(dMaxVal < vecLayerScore[iTopLayer])
                    break;
                    
                vecMatchParameter.push_back(s_MatchParameter(
                    cv::Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY),
                    dMaxVal, dAngle));
            }
        }
        else
        {
            double dMaxVal;
            cv::Point ptMaxLoc;
            cv::minMaxLoc(matResult, nullptr, &dMaxVal, nullptr, &ptMaxLoc);

            if(dMaxVal < vecLayerScore[iTopLayer])
                continue;

            vecMatchParameter.push_back(s_MatchParameter(
                cv::Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY),
                dMaxVal, dAngle));

            for(int j = 0; j < m_iMaxPos + MATCH_CANDIDATE_NUM - 1; j++)
            {
                ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc,
                                       pTemplData->vecPyramid[iTopLayer].size(),
                                       dMaxVal, m_dMaxOverlap);
                                       
                if(dMaxVal < vecLayerScore[iTopLayer])
                    break;
                    
                vecMatchParameter.push_back(s_MatchParameter(
                    cv::Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY),
                    dMaxVal, dAngle));
            }
        }
    }

    // 按分数排序
    sort(vecMatchParameter.begin(), vecMatchParameter.end(), compareScoreBig2Small);

    int iMatchSize = (int)vecMatchParameter.size();
    int iDstW = pTemplData->vecPyramid[iTopLayer].cols;
    int iDstH = pTemplData->vecPyramid[iTopLayer].rows;

    // 停止在第一层
    int iStopLayer = m_bStopLayer1 ? 1 : 0;
    std::vector<s_MatchParameter> vecAllResult;

    // Process each match result
    for(const auto& param : vecMatchParameter)
    {
        double dRAngle = -param.dMatchAngle * D2R;
        cv::Point2f ptLT = ptRotatePt2f(cv::Point2f(param.pt), ptCenter, dRAngle);

        double dAngleStep = atan(2.0 / max(iDstW, iDstH)) * R2D;
        double dAngleStart = param.dMatchAngle - dAngleStep;
        double dAngleEnd = param.dMatchAngle + dAngleStep;

        if(iTopLayer <= iStopLayer)
        {
            s_MatchParameter newParam = param;
            newParam.pt = cv::Point2d(ptLT * ((iTopLayer == 0) ? 1 : 2));
            vecAllResult.push_back(newParam);
        }
        else
        {
            s_MatchParameter currentParam = param;
            cv::Point2f currentPtLT = ptLT;

            // Refine through pyramid levels
            for(int iLayer = iTopLayer - 1; iLayer >= iStopLayer; iLayer--)
            {
                // 搜索角度
                dAngleStep = atan(2.0 / max(pTemplData->vecPyramid[iLayer].cols,
                                          pTemplData->vecPyramid[iLayer].rows)) * R2D;

                std::vector<double> vecRefineAngles;
                double dMatchedAngle = currentParam.dMatchAngle;

                if(m_bToleranceRange)
                {
                    for(int i = -1; i <= 1; i++)
                        vecRefineAngles.push_back(dMatchedAngle + dAngleStep * i);
                }
                else
                {
                    if(m_dToleranceAngle < VISION_TOLERANCE)
                        vecRefineAngles.push_back(0.0);
                    else
                        for(int i = -1; i <= 1; i++)
                            vecRefineAngles.push_back(dMatchedAngle + dAngleStep * i);
                }

                cv::Point2f ptSrcCenter((vecMatSrcPyr[iLayer].cols - 1) / 2.0f,
                                      (vecMatSrcPyr[iLayer].rows - 1) / 2.0f);

                std::vector<s_MatchParameter> vecNewMatchParameter(vecRefineAngles.size());
                int iMaxScoreIndex = 0;
                double dBigValue = -1;

                // Try each angle
                for(size_t j = 0; j < vecRefineAngles.size(); j++)
                {
                    cv::Mat matRotatedROI;
                    GetRotatedROI(vecMatSrcPyr[iLayer],
                                pTemplData->vecPyramid[iLayer].size(),
                                currentPtLT * 2,
                                vecRefineAngles[j],
                                matRotatedROI);

                    cv::Mat matResult;
                    MatchTemplate(matRotatedROI, pTemplData, matResult, iLayer, true);

                    double dMaxValue;
                    cv::Point ptMaxLoc;
                    cv::minMaxLoc(matResult, nullptr, &dMaxValue, nullptr, &ptMaxLoc);

                    vecNewMatchParameter[j] = s_MatchParameter(ptMaxLoc, dMaxValue, vecRefineAngles[j]);

                    if(vecNewMatchParameter[j].dMatchScore > dBigValue)
                    {
                        iMaxScoreIndex = j;
                        dBigValue = vecNewMatchParameter[j].dMatchScore;
                    }

                    // 次像素估计
                    if(ptMaxLoc.x == 0 || ptMaxLoc.y == 0 ||
                       ptMaxLoc.x == matResult.cols - 1 || 
                       ptMaxLoc.y == matResult.rows - 1)
                        vecNewMatchParameter[j].bPosOnBorder = true;

                    if(!vecNewMatchParameter[j].bPosOnBorder)
                    {
                        for(int y = -1; y <= 1; y++)
                            for(int x = -1; x <= 1; x++)
                                vecNewMatchParameter[j].vecResult[x+1][y+1] = 
                                    matResult.at<float>(ptMaxLoc + cv::Point(x, y));
                    }
                }

                if(vecNewMatchParameter[iMaxScoreIndex].dMatchScore < vecLayerScore[iLayer])
                    break;

                // 次像素估计
                if(m_bUseSubPixel && iLayer == 0 &&
                   !vecNewMatchParameter[iMaxScoreIndex].bPosOnBorder &&
                   iMaxScoreIndex != 0 && iMaxScoreIndex != 2)
                {
                    double dNewX = 0, dNewY = 0, dNewAngle = 0;
                    SubPixEsimation(&vecNewMatchParameter, &dNewX, &dNewY, &dNewAngle,
                                  dAngleStep, iMaxScoreIndex);

                    vecNewMatchParameter[iMaxScoreIndex].pt = cv::Point2d(dNewX, dNewY);
                    vecNewMatchParameter[iMaxScoreIndex].dMatchAngle = dNewAngle;
                }

                double dNewMatchAngle = vecNewMatchParameter[iMaxScoreIndex].dMatchAngle;

                // Update coordinates
                cv::Point2f ptPaddingLT = ptRotatePt2f(currentPtLT * 2, ptSrcCenter,
                                                      dNewMatchAngle * D2R) - cv::Point2f(3, 3);
                cv::Point2f pt(vecNewMatchParameter[iMaxScoreIndex].pt.x + ptPaddingLT.x,
                             vecNewMatchParameter[iMaxScoreIndex].pt.y + ptPaddingLT.y);
                pt = ptRotatePt2f(pt, ptSrcCenter, -dNewMatchAngle * D2R);

                if(iLayer == iStopLayer)
                {
                    vecNewMatchParameter[iMaxScoreIndex].pt = pt * (iStopLayer == 0 ? 1 : 2);
                    vecAllResult.push_back(vecNewMatchParameter[iMaxScoreIndex]);
                }
                else
                {
                    currentParam.dMatchAngle = dNewMatchAngle;
                    currentParam.dAngleStart = currentParam.dMatchAngle - dAngleStep / 2;
                    currentParam.dAngleEnd = currentParam.dMatchAngle + dAngleStep / 2;
                    currentPtLT = pt;
                }
            }
        }
    }

    // 按分数过滤结果
    FilterWithScore(&vecAllResult, m_dScore);

    // 最后过滤重叠区域
    iDstW = pTemplData->vecPyramid[iStopLayer].cols * (iStopLayer == 0 ? 1 : 2);
    iDstH = pTemplData->vecPyramid[iStopLayer].rows * (iStopLayer == 0 ? 1 : 2);

    for(auto& result : vecAllResult)
    {
        cv::Point2f ptLT, ptRT, ptRB, ptLB;
        double dRAngle = -result.dMatchAngle * D2R;
        
        ptLT = result.pt;
        ptRT = cv::Point2f(ptLT.x + iDstW * (float)cos(dRAngle), 
                          ptLT.y - iDstW * (float)sin(dRAngle));
        ptLB = cv::Point2f(ptLT.x + iDstH * (float)sin(dRAngle), 
                          ptLT.y + iDstH * (float)cos(dRAngle));
        ptRB = cv::Point2f(ptRT.x + iDstH * (float)sin(dRAngle), 
                          ptRT.y + iDstH * (float)cos(dRAngle));

        result.rectR = cv::RotatedRect(ptLT, ptRT, ptRB);
    }

    FilterWithRotatedRect(&vecAllResult, CV_TM_CCOEFF_NORMED, m_dMaxOverlap);

    // 按分数排序
    sort(vecAllResult.begin(), vecAllResult.end(), compareScoreBig2Small);

    m_vecSingleTargetData.clear();
    iMatchSize = (int)vecAllResult.size();
    
    if(vecAllResult.empty())
        return false;

    int iW = pTemplData->vecPyramid[0].cols;
    int iH = pTemplData->vecPyramid[0].rows;

    for(int i = 0; i < iMatchSize; i++)
    {
        s_SingleTargetMatch sstm;
        double dRAngle = -vecAllResult[i].dMatchAngle * D2R;

        sstm.ptLT = vecAllResult[i].pt;
        sstm.ptRT = cv::Point2d(sstm.ptLT.x + iW * cos(dRAngle),
                               sstm.ptLT.y - iW * sin(dRAngle));
        sstm.ptLB = cv::Point2d(sstm.ptLT.x + iH * sin(dRAngle),
                               sstm.ptLT.y + iH * cos(dRAngle));
        sstm.ptRB = cv::Point2d(sstm.ptRT.x + iH * sin(dRAngle),
                               sstm.ptRT.y + iH * cos(dRAngle));
        sstm.ptCenter = cv::Point2d((sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4,
                                   (sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
        sstm.dMatchedAngle = -vecAllResult[i].dMatchAngle;
        sstm.dMatchScore = vecAllResult[i].dMatchScore;

        if(sstm.dMatchedAngle < -180)
            sstm.dMatchedAngle += 360;
        if(sstm.dMatchedAngle > 180)
            sstm.dMatchedAngle -= 360;

        m_vecSingleTargetData.push_back(sstm);

        if(i + 1 == m_iMaxPos)
            break;
    }

    return (int)m_vecSingleTargetData.size();
}

void TemplateMatcher::LearnPattern()
{
    m_TemplData.clear();

    int iTopLayer = GetTopLayer(&m_matDst, (int)sqrt((double)m_iMinReduceArea));
    buildPyramid(m_matDst, m_TemplData.vecPyramid, iTopLayer);
    
    s_TemplData* templData = &m_TemplData;
    templData->iBorderColor = cv::mean(m_matDst)[0] < 128 ? 255 : 0;
    
    int iSize = templData->vecPyramid.size();
    templData->resize(iSize);

    for(int i = 0; i < iSize; i++)
    {
        double invArea = 1. / ((double)templData->vecPyramid[i].rows * templData->vecPyramid[i].cols);
        cv::Scalar templMean, templSdv;
        double templNorm = 0, templSum2 = 0;

        cv::meanStdDev(templData->vecPyramid[i], templMean, templSdv);
        templNorm = templSdv[0] * templSdv[0];

        if(templNorm < DBL_EPSILON)
        {
            templData->vecResultEqual1[i] = true;
        }
        
        templSum2 = templNorm + templMean[0] * templMean[0];
        templSum2 /= invArea;
        templNorm = std::sqrt(templNorm);
        templNorm /= std::sqrt(invArea);

        templData->vecInvArea[i] = invArea;
        templData->vecTemplMean[i] = templMean;
        templData->vecTemplNorm[i] = templNorm;
    }
    
    templData->bIsPatternLearned = true;
}

void TemplateMatcher::GetRotatedROI(cv::Mat& matSrc, cv::Size size, cv::Point2f ptLT, 
                                  double dAngle, cv::Mat& matROI)
{
    double dAngle_radian = dAngle * D2R;
    cv::Point2f ptC((matSrc.cols - 1) / 2.0f, (matSrc.rows - 1) / 2.0f);
    cv::Point2f ptLT_rotate = ptRotatePt2f(ptLT, ptC, dAngle_radian);
    cv::Size sizePadding(size.width + 6, size.height + 6);

    cv::Mat rMat = cv::getRotationMatrix2D(ptC, dAngle, 1);
    rMat.at<double>(0, 2) -= ptLT_rotate.x - 3;
    rMat.at<double>(1, 2) -= ptLT_rotate.y - 3;

    cv::warpAffine(matSrc, matROI, rMat, sizePadding);
}

bool TemplateMatcher::SubPixEsimation(std::vector<s_MatchParameter>* vec, double* dNewX,
                                    double* dNewY, double* dNewAngle, double dAngleStep,
                                    int iMaxScoreIndex)
{
    cv::Mat matA(27, 10, CV_64F);
    cv::Mat matZ(10, 1, CV_64F);
    cv::Mat matS(27, 1, CV_64F);

    double dX_maxScore = (*vec)[iMaxScoreIndex].pt.x;
    double dY_maxScore = (*vec)[iMaxScoreIndex].pt.y;
    double dTheata_maxScore = (*vec)[iMaxScoreIndex].dMatchAngle;
    int iRow = 0;

    for(int theta = 0; theta <= 2; theta++)
    {
        for(int y = -1; y <= 1; y++)
        {
            for(int x = -1; x <= 1; x++)
            {
                double dX = dX_maxScore + x;
                double dY = dY_maxScore + y;
                double dT = (dTheata_maxScore + (theta - 1) * dAngleStep) * D2R;

                matA.at<double>(iRow, 0) = dX * dX;
                matA.at<double>(iRow, 1) = dY * dY;
                matA.at<double>(iRow, 2) = dT * dT;
                matA.at<double>(iRow, 3) = dX * dY;
                matA.at<double>(iRow, 4) = dX * dT;
                matA.at<double>(iRow, 5) = dY * dT;
                matA.at<double>(iRow, 6) = dX;
                matA.at<double>(iRow, 7) = dY;
                matA.at<double>(iRow, 8) = dT;
                matA.at<double>(iRow, 9) = 1.0;
                matS.at<double>(iRow, 0) = (*vec)[iMaxScoreIndex + (theta - 1)].vecResult[x + 1][y + 1];
                iRow++;
            }
        }
    }

    matZ = (matA.t() * matA).inv() * matA.t() * matS;

    cv::Mat matK1 = (cv::Mat_<double>(3, 3) << 
        (2 * matZ.at<double>(0, 0)), matZ.at<double>(3, 0), matZ.at<double>(4, 0),
        matZ.at<double>(3, 0), (2 * matZ.at<double>(1, 0)), matZ.at<double>(5, 0),
        matZ.at<double>(4, 0), matZ.at<double>(5, 0), (2 * matZ.at<double>(2, 0)));

    cv::Mat matK2 = (cv::Mat_<double>(3, 1) << 
        -matZ.at<double>(6, 0),
        -matZ.at<double>(7, 0),
        -matZ.at<double>(8, 0));

    cv::Mat matDelta = matK1.inv() * matK2;

    *dNewX = matDelta.at<double>(0, 0);
    *dNewY = matDelta.at<double>(1, 0);
    *dNewAngle = matDelta.at<double>(2, 0) * R2D;

    return true;
}

void TemplateMatcher::MatchTemplate(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, bool bUseSIMD)
{
    if (m_bUseSIMD && bUseSIMD)
    {
        // 原有的SIMD实现保持不变
        matResult.create(matSrc.rows - pTemplData->vecPyramid[iLayer].rows + 1,
            matSrc.cols - pTemplData->vecPyramid[iLayer].cols + 1, CV_32FC1);
        matResult.setTo(0);
        cv::Mat& matTemplate = pTemplData->vecPyramid[iLayer];

        int t_r_end = matTemplate.rows;
        for (int r = 0; r < matResult.rows; r++)
        {
            float* r_matResult = matResult.ptr<float>(r);
            uchar* r_source = matSrc.ptr<uchar>(r);
            uchar* r_template, * r_sub_source;
            for (int c = 0; c < matResult.cols; ++c, ++r_matResult, ++r_source)
            {
                r_template = matTemplate.ptr<uchar>();
                r_sub_source = r_source;
                for (int t_r = 0; t_r < t_r_end; ++t_r, r_sub_source += matSrc.cols, r_template += matTemplate.cols)
                {
                    *r_matResult = float(*r_matResult +
                        match_utils::IM_Conv_SIMD(r_template, r_sub_source, matTemplate.cols));
                }
            }
        }

        // [新增] SIMD结果归一化
        double dScale = 1.0 / (255.0 * matTemplate.cols * matTemplate.rows);
        matResult *= dScale;
    }
    else
    {
        // OpenCV实现保持不变
        cv::matchTemplate(matSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCORR);
    }

    // 归一化过程保持不变
    if (!pTemplData->vecResultEqual1[iLayer])
    {
        CCOEFF_Denominator(matSrc, pTemplData, matResult, iLayer);
    }
    else
    {
        matResult = cv::Scalar::all(1);
    }

    // [新增] 确保结果在0-1范围内
    double minVal, maxVal;
    cv::minMaxLoc(matResult, &minVal, &maxVal);
    if (maxVal > 1.0 + DBL_EPSILON)
    {
        matResult /= maxVal;
    }
}

void TemplateMatcher::CCOEFF_Denominator(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer)
{
    if (pTemplData->vecResultEqual1[iLayer])
    {
        matResult = cv::Scalar::all(1);
        return;
    }

    cv::Mat sum, sqsum;
    cv::integral(matSrc, sum, sqsum, CV_64F);

    double templMean = pTemplData->vecTemplMean[iLayer][0];
    double templNorm = pTemplData->vecTemplNorm[iLayer];
    double invArea = pTemplData->vecInvArea[iLayer];

    int h = pTemplData->vecPyramid[iLayer].rows;
    int w = pTemplData->vecPyramid[iLayer].cols;

    for (int i = 0; i < matResult.rows; i++)
    {
        float* rrow = matResult.ptr<float>(i);
        for (int j = 0; j < matResult.cols; j++)
        {
            double num = rrow[j];

            // Get window sum and square sum
            double wndSum = sum.at<double>(i + h, j + w) -
                sum.at<double>(i + h, j) -
                sum.at<double>(i, j + w) +
                sum.at<double>(i, j);

            double wndSqSum = sqsum.at<double>(i + h, j + w) -
                sqsum.at<double>(i + h, j) -
                sqsum.at<double>(i, j + w) +
                sqsum.at<double>(i, j);

            // Calculate window mean and norm
            double wndMean = wndSum * invArea;
            double diff2 = std::max(wndSqSum - wndMean * wndSum, 0.0);

            double wndNorm;
            if (diff2 <= std::min(0.5, 10 * DBL_EPSILON * wndSqSum))
                wndNorm = 0;  // Avoid rounding errors
            else
                wndNorm = std::sqrt(diff2);

            // Normalize correlation score
            num = (num - wndMean * templMean * (w * h));
            double den = wndNorm * templNorm;

            if (den < DBL_EPSILON)
                rrow[j] = 0;
            else
                rrow[j] = (float)(num / den);

            // Clamp to [-1,1] range
            rrow[j] = std::min(std::max(rrow[j], -1.0f), 1.0f);
        }
    }
}

cv::Point2f TemplateMatcher::ptRotatePt2f(cv::Point2f ptInput, cv::Point2f ptOrg, double dAngle)
{
    double dWidth = ptOrg.x * 2;
    double dHeight = ptOrg.y * 2;
    double dY1 = dHeight - ptInput.y;
    double dY2 = dHeight - ptOrg.y;

    double dX = (ptInput.x - ptOrg.x) * cos(dAngle) - 
                (dY1 - ptOrg.y) * sin(dAngle) + ptOrg.x;
    double dY = (ptInput.x - ptOrg.x) * sin(dAngle) + 
                (dY1 - ptOrg.y) * cos(dAngle) + dY2;

    dY = -dY + dHeight;
    return cv::Point2f((float)dX, (float)dY);
}

// Implementation of remaining utility functions...
// DrawDashLine, DrawMarkCross, etc.
// Utility functions implementation
void TemplateMatcher::DrawDashLine(cv::Mat& matDraw, cv::Point ptStart, cv::Point ptEnd, 
                                 cv::Scalar color1, cv::Scalar color2)
{
    cv::LineIterator it(matDraw, ptStart, ptEnd, 8);
    for(int i = 0; i < it.count; i++, ++it)
    {
        if(i % 3 == 0)
        {
            (*it)[0] = (uchar)color2[0];
            (*it)[1] = (uchar)color2[1];
            (*it)[2] = (uchar)color2[2];
        }
        else
        {
            (*it)[0] = (uchar)color1[0];
            (*it)[1] = (uchar)color1[1];
            (*it)[2] = (uchar)color1[2];
        }
    }
}

void TemplateMatcher::DrawMarkCross(cv::Mat& matDraw, int iX, int iY, int iLength,
                                  cv::Scalar color, int iThickness)
{
    if(matDraw.empty())
        return;
    
    cv::Point ptC(iX, iY);
    cv::line(matDraw, ptC - cv::Point(iLength, 0), ptC + cv::Point(iLength, 0),
             color, iThickness);
    cv::line(matDraw, ptC - cv::Point(0, iLength), ptC + cv::Point(0, iLength),
             color, iThickness);
}

cv::Size TemplateMatcher::GetBestRotationSize(cv::Size sizeSrc, cv::Size sizeDst, double dRAngle)
{
    double dRAngle_radian = dRAngle * D2R;
    cv::Point ptLT(0, 0), ptLB(0, sizeSrc.height - 1);
    cv::Point ptRB(sizeSrc.width - 1, sizeSrc.height - 1), ptRT(sizeSrc.width - 1, 0);
    cv::Point2f ptCenter((sizeSrc.width - 1) / 2.0f, (sizeSrc.height - 1) / 2.0f);

    cv::Point2f ptLT_R = ptRotatePt2f(cv::Point2f(ptLT), ptCenter, dRAngle_radian);
    cv::Point2f ptLB_R = ptRotatePt2f(cv::Point2f(ptLB), ptCenter, dRAngle_radian);
    cv::Point2f ptRB_R = ptRotatePt2f(cv::Point2f(ptRB), ptCenter, dRAngle_radian);
    cv::Point2f ptRT_R = ptRotatePt2f(cv::Point2f(ptRT), ptCenter, dRAngle_radian);

    float fTopY = std::max(std::max(ptLT_R.y, ptLB_R.y), std::max(ptRB_R.y, ptRT_R.y));
    float fBottomY = std::min(std::min(ptLT_R.y, ptLB_R.y), std::min(ptRB_R.y, ptRT_R.y));
    float fRightX = std::max(std::max(ptLT_R.x, ptLB_R.x), std::max(ptRB_R.x, ptRT_R.x));
    float fLeftX = std::min(std::min(ptLT_R.x, ptLB_R.x), std::min(ptRB_R.x, ptRT_R.x));

    if(dRAngle > 360)
        dRAngle -= 360;
    else if(dRAngle < 0)
        dRAngle += 360;

    if(fabs(fabs(dRAngle) - 90) < VISION_TOLERANCE || 
       fabs(fabs(dRAngle) - 270) < VISION_TOLERANCE)
    {
        return cv::Size(sizeSrc.height, sizeSrc.width);
    }
    else if(fabs(dRAngle) < VISION_TOLERANCE || 
            fabs(fabs(dRAngle) - 180) < VISION_TOLERANCE)
    {
        return sizeSrc;
    }

    double dAngle = dRAngle;
    if(dAngle > 0 && dAngle < 90)
    {
        ;
    }
    else if(dAngle > 90 && dAngle < 180)
    {
        dAngle -= 90;
    }
    else if(dAngle > 180 && dAngle < 270)
    {
        dAngle -= 180;
    }
    else if(dAngle > 270 && dAngle < 360)
    {
        dAngle -= 270;
    }

    float fH1 = sizeDst.width * sin(dAngle * D2R) * cos(dAngle * D2R);
    float fH2 = sizeDst.height * sin(dAngle * D2R) * cos(dAngle * D2R);

    int iHalfHeight = (int)ceil(fTopY - ptCenter.y - fH1);
    int iHalfWidth = (int)ceil(fRightX - ptCenter.x - fH2);

    cv::Size sizeRet(iHalfWidth * 2, iHalfHeight * 2);

    bool bWrongSize = (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height) ||
                     (sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height) ||
                     (sizeDst.area() > sizeRet.area());

    if(bWrongSize)
        sizeRet = cv::Size(int(fRightX - fLeftX + 0.5), int(fTopY - fBottomY + 0.5));

    return sizeRet;
}

cv::Point TemplateMatcher::GetNextMaxLoc(cv::Mat& matResult, cv::Point ptMaxLoc,
                                       cv::Size sizeTemplate, double& dMaxValue, 
                                       double dMaxOverlap)
{
    int iStartX = ptMaxLoc.x - int(sizeTemplate.width * (1 - dMaxOverlap));
    int iStartY = ptMaxLoc.y - int(sizeTemplate.height * (1 - dMaxOverlap));

    cv::Rect rectIgnore(iStartX, iStartY,
                       int(2 * sizeTemplate.width * (1 - dMaxOverlap)),
                       int(2 * sizeTemplate.height * (1 - dMaxOverlap)));

    cv::rectangle(matResult, rectIgnore, cv::Scalar(-1), CV_FILLED);

    cv::Point ptNewMaxLoc;
    cv::minMaxLoc(matResult, nullptr, &dMaxValue, nullptr, &ptNewMaxLoc);
    return ptNewMaxLoc;
}

void TemplateMatcher::FilterWithScore(std::vector<s_MatchParameter>* vec, double dScore)
{
    std::sort(vec->begin(), vec->end(), compareScoreBig2Small);
    
    int iSize = vec->size(), iIndexDelete = iSize + 1;
    for(int i = 0; i < iSize; i++)
    {
        if((*vec)[i].dMatchScore < dScore)
        {
            iIndexDelete = i;
            break;
        }
    }

    if(iIndexDelete == iSize + 1)
        return;

    vec->erase(vec->begin() + iIndexDelete, vec->end());
}

void TemplateMatcher::FilterWithRotatedRect(std::vector<s_MatchParameter>* vec,
                                          int iMethod, double dMaxOverLap)
{
    int iMatchSize = (int)vec->size();
    cv::RotatedRect rect1, rect2;

    for(int i = 0; i < iMatchSize - 1; i++)
    {
        if(vec->at(i).bDelete)
            continue;

        for(int j = i + 1; j < iMatchSize; j++)
        {
            if(vec->at(j).bDelete)
                continue;

            rect1 = vec->at(i).rectR;
            rect2 = vec->at(j).rectR;

            std::vector<cv::Point2f> vecInterSec;
            std::vector<cv::Point2f> vertices;
            int iInterSecType = rotatedRectangleIntersection(rect1, rect2, vecInterSec);

            if(iInterSecType == cv::INTERSECT_NONE)
                continue;
            else if(iInterSecType == cv::INTERSECT_FULL)
            {
                int iDeleteIndex = (iMethod == CV_TM_SQDIFF) ? 
                    (vec->at(i).dMatchScore <= vec->at(j).dMatchScore ? j : i) :
                    (vec->at(i).dMatchScore >= vec->at(j).dMatchScore ? j : i);
                vec->at(iDeleteIndex).bDelete = true;
            }
            else
            {
                if(vecInterSec.size() < 3)
                    continue;
                
                int iDeleteIndex;
                std::vector<cv::Point2f> hull;
                cv::convexHull(vecInterSec, hull);
                double dArea = cv::contourArea(hull);
                double dRatio = dArea / rect1.size.area();

                if(dRatio > dMaxOverLap)
                {
                    iDeleteIndex = (iMethod == CV_TM_SQDIFF) ?
                        (vec->at(i).dMatchScore <= vec->at(j).dMatchScore ? j : i) :
                        (vec->at(i).dMatchScore >= vec->at(j).dMatchScore ? j : i);
                    vec->at(iDeleteIndex).bDelete = true;
                }
            }
        }
    }

    std::vector<s_MatchParameter>::iterator it = vec->begin();
    while(it != vec->end())
    {
        if((*it).bDelete)
            it = vec->erase(it);
        else
            ++it;
    }
}

// BlockMax implementation
TemplateMatcher::s_BlockMax::s_BlockMax(cv::Mat& src, cv::Size sizeTemplate)
    : matSrc(src)
{
    int iBlockW = sizeTemplate.width * 2;
    int iBlockH = sizeTemplate.height * 2;

    int iCol = src.cols / iBlockW;
    bool bHResidue = src.cols % iBlockW != 0;

    int iRow = src.rows / iBlockH;
    bool bVResidue = src.rows % iBlockH != 0;

    if(iCol == 0 || iRow == 0)
    {
        vecBlock.clear();
        return;
    }

    vecBlock.resize(iCol * iRow);
    int iCount = 0;
    for(int y = 0; y < iRow; y++)
    {
        for(int x = 0; x < iCol; x++)
        {
            Block block;
            block.rect = cv::Rect(x * iBlockW, y * iBlockH, iBlockW, iBlockH);
            cv::minMaxLoc(src(block.rect), nullptr, &block.dMax, nullptr, &block.ptMaxLoc);
            block.ptMaxLoc += cv::Point(block.rect.x, block.rect.y);
            vecBlock[iCount++] = block;
        }
    }

    if(bHResidue && bVResidue)
    {
        Block blockRight;
        blockRight.rect = cv::Rect(iCol * iBlockW, 0, 
                                 src.cols - iCol * iBlockW, src.rows);
        cv::minMaxLoc(src(blockRight.rect), nullptr, &blockRight.dMax,
                     nullptr, &blockRight.ptMaxLoc);
        blockRight.ptMaxLoc += cv::Point(blockRight.rect.x, blockRight.rect.y);
        vecBlock.push_back(blockRight);

        Block blockBottom;
        blockBottom.rect = cv::Rect(0, iRow * iBlockH,
                                  iCol * iBlockW, src.rows - iRow * iBlockH);
        cv::minMaxLoc(src(blockBottom.rect), nullptr, &blockBottom.dMax,
                     nullptr, &blockBottom.ptMaxLoc);
        blockBottom.ptMaxLoc += cv::Point(blockBottom.rect.x, blockBottom.rect.y);
        vecBlock.push_back(blockBottom);
    }
    else if(bHResidue)
    {
        Block blockRight;
        blockRight.rect = cv::Rect(iCol * iBlockW, 0,
                                 src.cols - iCol * iBlockW, src.rows);
        cv::minMaxLoc(src(blockRight.rect), nullptr, &blockRight.dMax,
                     nullptr, &blockRight.ptMaxLoc);
        blockRight.ptMaxLoc += cv::Point(blockRight.rect.x, blockRight.rect.y);
        vecBlock.push_back(blockRight);
    }
    else if(bVResidue)
    {
        Block blockBottom;
        blockBottom.rect = cv::Rect(0, iRow * iBlockH,
                                  src.cols, src.rows - iRow * iBlockH);
        cv::minMaxLoc(src(blockBottom.rect), nullptr, &blockBottom.dMax,
                     nullptr, &blockBottom.ptMaxLoc);
        blockBottom.ptMaxLoc += cv::Point(blockBottom.rect.x, blockBottom.rect.y);
        vecBlock.push_back(blockBottom);
    }
}

void TemplateMatcher::s_BlockMax::UpdateMax(cv::Rect rectIgnore)
{
    if(vecBlock.empty())
        return;

    for(auto& block : vecBlock)
    {
        cv::Rect rectIntersec = rectIgnore & block.rect;
        if(rectIntersec.width == 0 && rectIntersec.height == 0)
            continue;

        cv::minMaxLoc(matSrc(block.rect), nullptr, &block.dMax,
                     nullptr, &block.ptMaxLoc);
        block.ptMaxLoc += cv::Point(block.rect.x, block.rect.y);
    }
}

void TemplateMatcher::s_BlockMax::GetMaxValueLoc(double& dMax, cv::Point& ptMaxLoc)
{
    if(vecBlock.empty())
    {
        cv::minMaxLoc(matSrc, nullptr, &dMax, nullptr, &ptMaxLoc);
        return;
    }

    int iIndex = 0;
    dMax = vecBlock[0].dMax;
    for(int i = 1; i < (int)vecBlock.size(); i++)
    {
        if(vecBlock[i].dMax >= dMax)
        {
            iIndex = i;
            dMax = vecBlock[i].dMax;
        }
    }
    ptMaxLoc = vecBlock[iIndex].ptMaxLoc;
}

int TemplateMatcher::GetTopLayer(cv::Mat* matTempl, int iMinDstLength)
{
    int iTopLayer = 0;
    int iMinReduceArea = iMinDstLength * iMinDstLength;
    int iArea = matTempl->cols * matTempl->rows;

    while (iArea > iMinReduceArea)
    {
        iArea /= 4;
        iTopLayer++;
    }

    return iTopLayer;
}

cv::Point TemplateMatcher::GetNextMaxLoc(cv::Mat& matResult, cv::Point ptMaxLoc,
    cv::Size sizeTemplate, double& dMaxValue, double dMaxOverlap,
    s_BlockMax& blockMax)
{
    int iStartX = int(ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap));
    int iStartY = int(ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap));
    cv::Rect rectIgnore(iStartX, iStartY,
        int(2 * sizeTemplate.width * (1 - dMaxOverlap)),
        int(2 * sizeTemplate.height * (1 - dMaxOverlap)));

    rectangle(matResult, rectIgnore, cv::Scalar(-1), CV_FILLED);
    blockMax.UpdateMax(rectIgnore);
    cv::Point ptReturn;
    blockMax.GetMaxValueLoc(dMaxValue, ptReturn);
    return ptReturn;
}