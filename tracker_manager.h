#ifndef EYEDID_CPP_SAMPLE_TRACKER_MANAGER_H_
#define EYEDID_CPP_SAMPLE_TRACKER_MANAGER_H_

#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <vector>
#include <deque>

#include "seeso/gaze_tracker.h"
#include "seeso/util/display.h"

#include "opencv2/opencv.hpp"
#include "kalman_filter.h"

#include "simple_signal.h"

namespace sample {

    class TrackerManager :
        public seeso::IGazeCallback,
        public seeso::ICalibrationCallback {
    public:
        TrackerManager() = default;

        bool initialize(const std::string& license_key, const SeeSoStatusModuleOptions& options);

        void setDefaultCameraToDisplayConverter(const seeso::DisplayInfo& display_info);
        
        void setViewSize(int width, int height);

        bool addFrame(std::int64_t timestamp, const cv::Mat& frame);

        void startFullWindowCalibration(SeeSoCalibrationPointNum target_num, SeeSoCalibrationAccuracy accuracy);

        void setWholeScreenToAttentionRegion(const seeso::DisplayInfo& display_info);

        void calibrateOffset(int screen_center_x, int screen_center_y, int current_gaze_x, int current_gaze_y);
        
        // Multi-point calibration: calibrate at 5 points (4 corners + center)
        void startMultiPointCalibration(int view_width, int view_height);
        bool isCalibratingMultiPoint() const { return multi_point_calibrating_; }
        int getCurrentCalibrationPoint() const { return current_calibration_point_; }
        
        // Get current raw gaze coordinates (before offset correction)
        std::pair<int, int> getCurrentRawGaze() const;

        signal<void(int, int, bool)> on_gaze_;
        signal<void(float)> on_calib_progress_;
        signal<void(int, int)> on_calib_next_point_;
        signal<void()> on_calib_start_;
        signal<void(const std::vector<float>&)> on_calib_finish_;

        std::string window_name_;

    private:
        void OnGaze(uint64_t timestamp, float x, float y, float fixation_x, float fixation_y,
            float left_openness, float right_openness,
            SeeSoTrackingState tracking_state, SeeSoEyeMovementState eye_movement_state) override;
        void OnFace(uint64_t timestamp, float score, float left, float top, float right, float bottom,
            float pitch, float yaw, float roll, float center_x, float center_y, float center_z);
        void OnAttention(float score);
        void OnBlink(uint64_t timestamp, bool isBlinkLeft, bool isBlinkRight, bool isBlink,
            float leftOpenness, float rightOpenness);

        void OnDrowsiness(uint64_t timestamp, bool isDrowsiness, float intensity);

        void OnCalibrationProgress(float progress) override;
        void OnCalibrationNextPoint(float next_point_x, float next_point_y) override;
        void OnCalibrationFinish(const std::vector<float>& calib_data) override;
        void OnCalibrationCancel(const std::vector<float>& calib_data);

        seeso::GazeTracker gaze_tracker_;
        std::future<void> delayed_calibration_;
        std::atomic_bool calibrating_;

        // Kalman filter for gaze tracking stabilization
        KalmanFilter2D kalman_filter_;

        float offset_x_;
        float offset_y_;
        bool offset_calibrated_;
        
        // Multi-point calibration data
        struct CalibrationPoint {
            int target_x, target_y;  // Target screen position
            float offset_x, offset_y;  // Calculated offset
            bool calibrated;
            std::vector<std::pair<int, int>> samples;  // Collected samples
        };
        std::vector<CalibrationPoint> calibration_points_;
        bool multi_point_calibrated_;
        bool multi_point_calibrating_;
        int current_calibration_point_;
        std::chrono::steady_clock::time_point calibration_start_time_;
        std::chrono::steady_clock::time_point last_sample_time_;
        static const int CALIBRATION_DURATION_MS = 5000;  // 5 seconds per point
        static const int CALIBRATION_SAMPLE_INTERVAL_MS = 100;  // Sample every 100ms
        
        // Store current raw gaze coordinates (before offset correction)
        mutable int current_raw_gaze_x_;
        mutable int current_raw_gaze_y_;
        
        // View dimensions for mirroring
        int view_width_;
        int view_height_;
        
        // Calculate interpolated offset based on current gaze position
        std::pair<float, float> getInterpolatedOffset(float gaze_x, float gaze_y) const;

    };

} // namespace sample

#endif // EYEDID_CPP_SAMPLE_TRACKER_MANAGER_H_
