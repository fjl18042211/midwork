#include <iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int redThre = 115;
int saturationTh = 45;
Mat CheckFire(Mat &inImg);
void DrawFire(Mat &inputImg, Mat foreImg);
Mat CheckFire(Mat &inputImg)
{
	Mat fireImg;
	fireImg.create(inputImg.size(), CV_8UC1);
	Mat multiRGB[3];
	int a = inputImg.channels();
	split(inputImg, multiRGB); //将图片拆分成R,G,B,三通道的颜色  

	for (int i = 0; i < inputImg.rows; i++)
	{
		for (int j = 0; j < inputImg.cols; j++)
		{
			float B, G, R;
			B = multiRGB[0].at<uchar>(i, j); 
			G = multiRGB[1].at<uchar>(i, j);
			R = multiRGB[2].at<uchar>(i, j);
			float maxValue = max(max(B, G), R);
			float minValue = min(min(B, G), R);
			//与HSI中S分量的计算公式
			double S = (1 - 3.0*minValue / (R + G + B));//
			if (R > redThre &&R >= G && G >= B && S > ((255 - R) * saturationTh / redThre) && R - G - B > -100)
			{
				fireImg.at<uchar>(i, j) = 255;
			}
			else
			{
				fireImg.at<uchar>(i, j) = 0;
			}
		}
	}
	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7), Point(3, 3));
	GaussianBlur(fireImg, fireImg, Size(5, 5), 0, 0);
	dilate(fireImg, fireImg, element);
	DrawFire(inputImg, fireImg);
	return fireImg;
}

void DrawFire(Mat &inputImg, Mat foreImg)
{
	vector<vector<Point>> contours;//保存轮廓提取后的点集及拓扑关系  

	findContours(foreImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	Mat result0;
	Scalar holeColor;
	Scalar externalColor;

	vector<vector<Point> >::iterator iter = contours.begin();
	for (; iter != contours.end(); )
	{
		Rect rect = boundingRect(*iter);
		float radius;
		Point2f center;
	

		if (rect.area() > 0)
		{
			rectangle(inputImg, rect, Scalar(0, 255, 0));
			++iter;
		}
		else
		{
			iter = contours.erase(iter);
		}
	}

	
}
std::vector<cv::Point> getPoints(cv::Mat &image, int Value)
{
	std::vector<cv::Point> points;
	for (int j = 30; j < 230; j++)
	{
		uchar* data = image.ptr<uchar>(j);
		for (int i = 170; i < 430; i++)
		{
			if (data[i] == Value)
			{
				points.push_back(cv::Point(i, j));
			}
		}
	}
	return points;
}
bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
	//Number of key points
	int N = key_point.size();
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					pow(key_point[k].x, i + j);
			}
		}
	}
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				pow(key_point[k].x, i) * key_point[k].y;
		}
	}
	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}
void DrawWatercrack(Mat &inputImg, Mat resultImg) {
	Mat clone = resultImg.clone();
	std::vector<cv::Point> points = getPoints(inputImg, 255);
	for (int i = 0; i < points.size(); i++)
	{
		cv::circle(inputImg, points[i], 5, cv::Scalar(0, 0, 255), 2, 8, 0);
	}
	cv::Mat A;
	polynomial_curve_fit(points, 2, A);
	std::vector<cv::Point> points_fitted;
	for (int x = 0; x < inputImg.cols; x++)
	{
		double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
			A.at<double>(2, 0)*std::pow(x, 2);
		points_fitted.push_back(cv::Point(x, y));
	}
	cv::polylines(resultImg, points_fitted, false, cv::Scalar(0, 255, 255), 2, 8, 0);
	for (int i = 0; i < resultImg.rows; i++) {
		uchar* data = resultImg.ptr<uchar>(i);
		for (int j = 0; j < 515; j++) {
			if (data[j] != clone.at<uchar>(i, j))//不需要的的曲线部分恢复原来的像素
				resultImg.at<uchar>(i, j) = clone.at<uchar>(i, j);
		}
	}
}
int main()
{
	VideoCapture capture("fire.mp4");
	int cnt = 0;
	Mat frame, bgMat, bny_subMat, subMat;
	Mat clone = frame.clone();
	while (capture.read(frame))
	{
		capture >> frame;
		Mat clone = frame.clone();
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		if (cnt == 0)
		{
			frame.copyTo(bgMat);

		}
		else if(cnt<40){
			CheckFire(clone);
			imshow("result",clone );
			waitKey(1);
		}
		else  
		{
			absdiff(frame, bgMat, subMat);
			for (int i = 0; i < subMat.rows; i++) {
				for (int j = 0; j < 180; j++) {
					subMat.at<uchar>(i, j) = 0;
				}
			}
			for (int i = 0; i < 100; i++) {
				for (int j = 300; j < 480; j++) {
					subMat.at<uchar>(i, j) = 0;
				}
			}
			for (int i = 120; i < 265; i++) {
				for (int j = 175; j < 270; j++) {
					subMat.at<uchar>(i, j) = 0;
				}
			}
			threshold(subMat, bny_subMat, 110, 255, THRESH_BINARY);
			CheckFire(clone);
			DrawWatercrack(bny_subMat, clone);
			imshow("result", clone);
			waitKey(30);
		}
		cnt++;
	}
	return 0;
}