#ifndef EYEDID_CPP_SAMPLE_KALMAN_FILTER_H_
#define EYEDID_CPP_SAMPLE_KALMAN_FILTER_H_

#include <opencv2/opencv.hpp>

namespace sample {

/**
 * Kalman filter for 2D position tracking
 * State vector: [x, y, vx, vy] (position and velocity)
 * Measurement vector: [x, y] (gaze position)
 */
class KalmanFilter2D {
public:
    KalmanFilter2D();
    
    // Initialize Kalman filter
    void initialize(float initial_x, float initial_y);
    
    // Update filter with new measurement
    cv::Point2f update(float measured_x, float measured_y);
    
    // Get current predicted position
    cv::Point2f getCurrentPosition() const;
    
    // Reset filter
    void reset();
    
    // Adjust process noise and measurement noise
    void setProcessNoise(float noise);
    void setMeasurementNoise(float noise);

private:
    cv::KalmanFilter kf_;
    cv::Mat measurement_;
    bool initialized_;
    
    // Noise parameters
    float process_noise_;
    float measurement_noise_;
    
    void setupKalmanFilter();
};

} // namespace sample

#endif // EYEDID_CPP_SAMPLE_KALMAN_FILTER_H_
