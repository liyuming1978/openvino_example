#include "poseestimate.h"

vector<cv::Point3d> MODEL_3D68 = {
	cv::Point3d(-73.39352417, -29.80143166, -47.66753387),
	cv::Point3d(-72.77501678, -10.94976616, -45.90940475),
	cv::Point3d(-70.533638  ,   7.92981815, -44.84257889),
	cv::Point3d(-66.85005951,  26.07427979, -43.14111328),
	cv::Point3d(-59.79018784,  42.56438828, -38.63529968),
	cv::Point3d(-48.36897278,  56.4810791 , -30.7506218),
	cv::Point3d(-34.12110138,  67.24699402, -18.45645332),
	cv::Point3d(-17.87541008,  75.0568924 ,  -3.60903502),
	cv::Point3d(0.098749  ,  77.06128693,   0.88169801),
	cv::Point3d(17.47703171,  74.75844574,  -5.18120098),
	cv::Point3d(32.64896774,  66.92902374, -19.17656326),
	cv::Point3d(46.37235641,  56.31138992, -30.77057076),
	cv::Point3d(57.34347916,  42.4191246 , -37.62862778),
	cv::Point3d(64.38848114,  25.45587921, -40.88631058),
	cv::Point3d(68.21203613,   6.99080515, -42.28144836),
	cv::Point3d(70.48640442, -11.66619301, -44.14256668),
	cv::Point3d(71.37582397, -30.36519051, -47.14042664),
	cv::Point3d(-61.11940765, -49.36160278, -14.25442219),
	cv::Point3d(-51.28758621, -58.76979446,  -7.26814699),
	cv::Point3d(-37.80479813, -61.99615479,  -0.44205099),
	cv::Point3d(-24.02275467, -61.03339767,   6.6065011),
	cv::Point3d(-11.63571262, -56.68675995,  11.96739769),
	cv::Point3d(12.05663586, -57.39103317,  12.05120373),
	cv::Point3d(25.10625648, -61.90218735,   7.31509781),
	cv::Point3d(38.33858871, -62.77771378,   1.02295303),
	cv::Point3d(51.19100571, -59.30234528,  -5.34943485),
	cv::Point3d(60.05385208, -50.19025421, -11.61574554),
	cv::Point3d(0.65394002, -42.19379044,  13.38083458),
	cv::Point3d(0.80480897, -30.99372101,  21.1508522),
	cv::Point3d(0.99220401, -19.94459534,  29.28403664),
	cv::Point3d(1.22678304,  -8.41454124,  36.94805908),
	cv::Point3d(-14.77247238,   2.59825492,  20.13200378),
	cv::Point3d(-7.1802392 ,   4.75158882,  23.53668404),
	cv::Point3d(0.55592   ,   6.56290007,  25.94444847),
	cv::Point3d(8.27249908,   4.66100502,  23.69574165),
	cv::Point3d(15.2143507 ,   2.6430459 ,  20.8581562),
	cv::Point3d(-46.0472908 , -37.47141266,  -7.03798914),
	cv::Point3d(-37.67468643, -42.73051071,  -3.02121711),
	cv::Point3d(-27.88385582, -42.71151733,  -1.35362899),
	cv::Point3d(-19.64826775, -36.75474167,   0.111088),
	cv::Point3d(-28.27296448, -35.13449478,   0.147273),
	cv::Point3d(-38.08241653, -34.91904449,  -1.47661197),
	cv::Point3d(19.26586723, -37.03230667,   0.66574597),
	cv::Point3d(27.89419174, -43.34244537,  -0.24766),
	cv::Point3d(37.43753052, -43.11082077,  -1.69643497),
	cv::Point3d(45.17080688, -38.08651352,  -4.89416313),
	cv::Point3d(38.19645309, -35.53202438,  -0.28296101),
	cv::Point3d(28.76498985, -35.48428726,   1.17267501),
	cv::Point3d(-28.9162674 ,  28.61271667,   2.24030995),
	cv::Point3d(-17.53319359,  22.17218781,  15.93433475),
	cv::Point3d(-6.68458986,  19.02905083,  22.61135483),
	cv::Point3d(0.381001  ,  20.72111893,  23.74843788),
	cv::Point3d(8.37544346,  19.03545952,  22.7219944),
	cv::Point3d(18.87661743,  22.39410973,  15.61067867),
	cv::Point3d(28.79441261,  28.07992363,   3.21739292),
	cv::Point3d(19.05757332,  36.29824829,  14.98799706),
	cv::Point3d(8.95637512,  39.63457489,  22.554245),
	cv::Point3d(0.381549  ,  40.39564514,  23.59162521),
	cv::Point3d(-7.428895  ,  39.83640671,  22.40610695),
	cv::Point3d(-18.16063309,  36.67789841,  15.12190723),
	cv::Point3d(-24.37748909,  28.67777061,   4.78568411),
	cv::Point3d(-6.89763308,  25.47597694,  20.89374161),
	cv::Point3d(0.34066299,  26.01426888,  22.22047806),
	cv::Point3d(8.44472218,  25.32619858,  21.02552032),
	cv::Point3d(24.47447395,  28.32300758,   5.71277618),
	cv::Point3d(8.4491663 ,  30.5962162 ,  20.67148972),
	cv::Point3d(0.205322  ,  31.40873718,  21.90366936),
	cv::Point3d(-7.19826603,  30.84487534,  20.328022),
};

vector<cv::Point3d> MODEL_3D = {
	cv::Point3d(0.0, 0.0, 0.0),                // Nose tip
	cv::Point3d(0.0, -330.0, -65.0),           // Chin
	cv::Point3d(-225.0, 170.0, -135.0),        // Left eye left corner
	cv::Point3d(225.0, 170.0, -135.0),         // Right eye right corner
	cv::Point3d(-150.0, -150.0, -125.0),       // Left mouth corner
	cv::Point3d(150.0, -150.0, -125.0),        // Right mouth corner
};

enum FeatureIndex {
	CHIN = 8,
	NOSE_TIP = 30,
	LEFT_EYE_CORNER = 36,
	RIGHT_EYE_CORNER = 45,
	MOUTH_LEFT = 48,
	MOUTH_RIGHT = 54,
};

PoseEstimate::PoseEstimate()
{
}


PoseEstimate::~PoseEstimate()
{
}

void PoseEstimate::computeRadian(cv::Mat & frame, FaceTracker::LandMarkInfo & l)
{
	cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
		frame.cols, 0, frame.cols / 2,
		0, frame.cols, frame.rows / 2,
		0, 0, 1);
	cv::Mat DIST_COEFFS = cv::Mat::zeros(4, 1, CV_64F);
	cv::Mat rotation, translation;

	vector<cv::Point2d> features;
	getFeatures(l, features);

	cv::solvePnP(MODEL_3D, features,
		cameraMatrix, DIST_COEFFS,
		rotation, translation);

	cv::Mat rotateMatrix;
	Rodrigues(rotation, rotateMatrix);
	rotationMatrixToEulerAngles(rotateMatrix,l);
}

void PoseEstimate::getFeatures(const FaceTracker::LandMarkInfo& li, vector<cv::Point2d>& features)
{
#if 0
	features.clear();
	for (int i = 0; i < LANDMARK_COUNT/2; i++) {
		cv::Point2d xy;
		xy.x = li.fmark[2 * i];
		xy.y = li.fmark[2 * i+1];
		features.push_back(xy);
	}
#else
	vector<FeatureIndex> FEATURE_INDICES = {
		NOSE_TIP,
		CHIN,
		LEFT_EYE_CORNER,
		RIGHT_EYE_CORNER,
		MOUTH_LEFT,
		MOUTH_RIGHT,
	};

	features.clear();
	for (int i = 0; i < FEATURE_INDICES.size(); i++) {
		int index = FEATURE_INDICES[i];
		cv::Point2d xy;
		xy.x = li.fmark[2 * index];
		xy.y = li.fmark[2 * index + 1];
		features.push_back(xy);
	}
#endif
}


bool PoseEstimate::isRotationMatrix(cv::Mat &R)
{
	cv::Mat Rt;
	cv::transpose(R, Rt);
	cv::Mat shouldBeIdentity = Rt * R;
	cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

	return  norm(I, shouldBeIdentity) < 1e-6;
}

void PoseEstimate::rotationMatrixToEulerAngles(cv::Mat &R, FaceTracker::LandMarkInfo & l)
{
	assert(isRotationMatrix(R));

	float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

	bool singular = sy < 1e-6; // If

	if (!singular) {
		l.y = atan2(-R.at<double>(2, 1), R.at<double>(2, 2));
		l.x = atan2(-R.at<double>(2, 0), sy);
		l.z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
	}
	else {
		l.y = atan2(R.at<double>(1, 2), R.at<double>(1, 1));
		l.x = atan2(-R.at<double>(2, 0), sy);
		l.z = 0;
	}
}
