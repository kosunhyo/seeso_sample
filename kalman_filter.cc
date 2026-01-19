#include "kalman_filter.h"
#include <cmath>

namespace sample {

KalmanFilter2D::KalmanFilter2D() 
    : initialized_(false), process_noise_(0.03f), measurement_noise_(0.3f) {
    setupKalmanFilter();
}

void KalmanFilter2D::setupKalmanFilter() {
    // State vector: [x, y, vx, vy] - 4 dimensions
    // Measurement vector: [x, y] - 2 dimensions
    // Control vector: none - 0 dimensions
    
    // Initialize for OpenCV version compatibility
    kf_.init(4, 2, 0, CV_32F);
    
    // Transition matrix (A): position and velocity update
    // x' = x + vx * dt
    // y' = y + vy * dt
    // vx' = vx
    // vy' = vy
    // dt = 1 (frame interval)
    cv::Mat transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,  // x' = x + vx
        0, 1, 0, 1,  // y' = y + vy
        0, 0, 1, 0,  // vx' = vx
        0, 0, 0, 1); // vy' = vy
    transitionMatrix.copyTo(kf_.transitionMatrix);
    
    // Measurement matrix (H): measure position only
    cv::Mat measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,  // measure x
        0, 1, 0, 0); // measure y
    measurementMatrix.copyTo(kf_.measurementMatrix);
    
    // Process covariance matrix (Q): process noise
    setProcessNoise(process_noise_);
    
    // Measurement covariance matrix (R): measurement noise
    setMeasurementNoise(measurement_noise_);
    
    // Posterior error covariance matrix (P): initial uncertainty
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1000));
    
    // Initialize measurement vector
    measurement_ = cv::Mat::zeros(2, 1, CV_32F);
}

void KalmanFilter2D::initialize(float initial_x, float initial_y) {
    // Set initial state
    kf_.statePre.at<float>(0) = initial_x;
    kf_.statePre.at<float>(1) = initial_y;
    kf_.statePre.at<float>(2) = 0.0f; // initial velocity vx
    kf_.statePre.at<float>(3) = 0.0f; // initial velocity vy
    
    kf_.statePost.at<float>(0) = initial_x;
    kf_.statePost.at<float>(1) = initial_y;
    kf_.statePost.at<float>(2) = 0.0f;
    kf_.statePost.at<float>(3) = 0.0f;
    
    initialized_ = true;
}

cv::Point2f KalmanFilter2D::update(float measured_x, float measured_y) {
    if (!initialized_) {
        initialize(measured_x, measured_y);
        return cv::Point2f(measured_x, measured_y);
    }
    
    // Prediction step
    cv::Mat prediction = kf_.predict();
    
    // Set measurement values
    measurement_.at<float>(0) = measured_x;
    measurement_.at<float>(1) = measured_y;
    
    // Update step (correct with measurement)
    cv::Mat estimated = kf_.correct(measurement_);
    
    // Return filtered position
    float filtered_x = estimated.at<float>(0);
    float filtered_y = estimated.at<float>(1);
    
    return cv::Point2f(filtered_x, filtered_y);
}

cv::Point2f KalmanFilter2D::getCurrentPosition() const {
    if (!initialized_) {
        return cv::Point2f(0.0f, 0.0f);
    }
    
    return cv::Point2f(
        kf_.statePost.at<float>(0),
        kf_.statePost.at<float>(1)
    );
}

void KalmanFilter2D::reset() {
    initialized_ = false;
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1000));
}

void KalmanFilter2D::setProcessNoise(float noise) {
    process_noise_ = noise;
    cv::Mat processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * noise;
    // Set smaller noise for velocity
    processNoiseCov.at<float>(2, 2) = noise * 0.1f;
    processNoiseCov.at<float>(3, 3) = noise * 0.1f;
    processNoiseCov.copyTo(kf_.processNoiseCov);
}

void KalmanFilter2D::setMeasurementNoise(float noise) {
    measurement_noise_ = noise;
    cv::Mat measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * noise;
    measurementNoiseCov.copyTo(kf_.measurementNoiseCov);
}

} // namespace sample
