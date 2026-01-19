#include "kalman_filter.h"
#include "tracker_manager.h"

#include <iostream>
#include <utility>
#include <vector>
#include <deque>
#include <cmath>
#include <chrono>
#include <algorithm>

#include "seeso/util/display.h"

namespace sample {

    static std::vector<float> getWindowRectWithPadding(const char* window_name, int padding = 30) {
        const auto window_rect = seeso::getWindowRect(window_name);
        return {
          static_cast<float>(window_rect.x + padding),
          static_cast<float>(window_rect.y + padding),
          static_cast<float>(window_rect.x + window_rect.width - padding),
          static_cast<float>(window_rect.y + window_rect.height - padding) };
    }

    void TrackerManager::OnGaze(uint64_t timestamp,
        float x, float y,
        float fixation_x, float fixation_y,
        float left_openness, float right_openness,
        SeeSoTrackingState tracking_state,
        SeeSoEyeMovementState eye_movement_state) {
        if (tracking_state != kSeeSoTrackingSuccess) {
            kalman_filter_.reset();
            on_gaze_(0, 0, false);
            return;
        }

        // Convert the gaze point(in display pixels) to the pixels of the OpenCV window
        auto winPos = seeso::getWindowPosition(window_name_);
        x -= static_cast<float>(winPos.x);
        y -= static_cast<float>(winPos.y);
        
        // Store raw gaze coordinates (before offset correction) for calibration
        current_raw_gaze_x_ = static_cast<int>(x);
        current_raw_gaze_y_ = static_cast<int>(y);

        // Focus correction: apply offset
        if (multi_point_calibrated_) {
            // Use multi-point calibration with bilinear interpolation
            auto interpolated_offset = getInterpolatedOffset(x, y);
            x -= interpolated_offset.first;
            y -= interpolated_offset.second;
        }
        else if (offset_calibrated_) {
            // Fallback to single-point calibration with distance-based scaling
            float screen_center_x = view_width_ / 2.0f;
            float screen_center_y = view_height_ / 2.0f;
            float dist_from_center = std::sqrt(
                (x - screen_center_x) * (x - screen_center_x) + 
                (y - screen_center_y) * (y - screen_center_y)
            );
            
            float max_dist = std::sqrt(screen_center_x * screen_center_x + screen_center_y * screen_center_y);
            float scale = 1.0f + (dist_from_center / max_dist) * 0.3f;
            x -= offset_x_ * scale;
            y -= offset_y_ * scale;
        }
        
        // Collect samples during multi-point calibration
        if (multi_point_calibrating_ && current_calibration_point_ >= 0 && 
            current_calibration_point_ < static_cast<int>(calibration_points_.size())) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - calibration_start_time_).count();
            
            if (elapsed < CALIBRATION_DURATION_MS) {
                // Sample at intervals
                auto sample_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_sample_time_).count();
                
                if (sample_elapsed >= CALIBRATION_SAMPLE_INTERVAL_MS) {
                    calibration_points_[current_calibration_point_].samples.push_back(
                        { current_raw_gaze_x_, current_raw_gaze_y_ });
                    last_sample_time_ = now;
                }
            }
            else {
                // Calculate average offset for this point
                auto& point = calibration_points_[current_calibration_point_];
                if (!point.samples.empty()) {
                    float avg_x = 0.0f, avg_y = 0.0f;
                    for (const auto& sample : point.samples) {
                        avg_x += static_cast<float>(sample.first);
                        avg_y += static_cast<float>(sample.second);
                    }
                    avg_x /= static_cast<float>(point.samples.size());
                    avg_y /= static_cast<float>(point.samples.size());
                    
                    point.offset_x = avg_x - static_cast<float>(point.target_x);
                    point.offset_y = avg_y - static_cast<float>(point.target_y);
                    point.calibrated = true;
                    
                    std::cout << "Calibration point " << (current_calibration_point_ + 1) << "/5 completed:\n";
                    std::cout << "  Target: (" << point.target_x << ", " << point.target_y << ")\n";
                    std::cout << "  Average gaze: (" << avg_x << ", " << avg_y << ")\n";
                    std::cout << "  Offset: (" << point.offset_x << ", " << point.offset_y << ")\n";
                }
                
                // Move to next point
                current_calibration_point_++;
                if (current_calibration_point_ >= static_cast<int>(calibration_points_.size())) {
                    // All points calibrated
                    multi_point_calibrating_ = false;
                    multi_point_calibrated_ = true;
                    std::cout << "Multi-point calibration completed!\n";
                }
                else {
                    calibration_start_time_ = std::chrono::steady_clock::now();
                    last_sample_time_ = calibration_start_time_;
                    std::cout << "Look at point " << (current_calibration_point_ + 1) << "/5 for 5 seconds...\n";
                }
            }
        }

        // Stabilize gaze position using Kalman filter
        cv::Point2f filtered_pos = kalman_filter_.update(x, y);
        
        on_gaze_(static_cast<int>(filtered_pos.x), static_cast<int>(filtered_pos.y), true);
    }

    void TrackerManager::OnFace(uint64_t timestamp,
        float score,
        float left,
        float top,
        float right,
        float bottom,
        float pitch,
        float yaw,
        float roll,
        float center_x,
        float center_y,
        float center_z) {
        std::cout << "Face Score: " << timestamp << ", " << score << '\n';
    }

    void TrackerManager::OnAttention(float score) {
        std::cout << "Attention: " << score << '\n';
    }

    void TrackerManager::OnBlink(uint64_t timestamp, bool isBlinkLeft, bool isBlinkRight, bool isBlink,
        float leftOpenness, float rightOpenness) {
        std::cout << "Blink: " << leftOpenness << ", " << rightOpenness << ", " << isBlinkLeft << ", " << isBlinkRight << '\n';
    }

    void TrackerManager::OnDrowsiness(uint64_t timestamp, bool isDrowsiness, float intensity) {
        std::cout << "Drowsiness: " << isDrowsiness << '\n';
    }

    void TrackerManager::OnCalibrationProgress(float progress) {
        on_calib_progress_(progress);
    }

    void TrackerManager::OnCalibrationNextPoint(float next_point_x, float next_point_y) {
        const auto winPos = seeso::getWindowPosition(window_name_);
        const auto x = static_cast<int>(next_point_x - static_cast<float>(winPos.x));
        const auto y = static_cast<int>(next_point_y - static_cast<float>(winPos.y));
        on_calib_next_point_(x, y);
        gaze_tracker_.startCollectSamples();
    }

    void TrackerManager::OnCalibrationFinish(const std::vector<float>& calib_data) {
        on_calib_finish_(calib_data);
        calibrating_.store(false, std::memory_order_release);
    }

    void TrackerManager::OnCalibrationCancel(const std::vector<float>& calib_data) {
        std::cout << "Calibration canceled\n";
    }

    bool TrackerManager::initialize(const std::string& license_key, const SeeSoStatusModuleOptions& options) {
        const auto code = gaze_tracker_.initialize(license_key, options);
        if (code != 0) {
            std::cerr << "Failed to authenticate (code: " << code << " )\n";
            return false;
        }

        // 초점 보정 변수 초기화
        offset_x_ = 0.0f;
        offset_y_ = 0.0f;
        offset_calibrated_ = false;
        multi_point_calibrated_ = false;
        multi_point_calibrating_ = false;
        current_calibration_point_ = -1;
        current_raw_gaze_x_ = 0;
        current_raw_gaze_y_ = 0;
        view_width_ = 0;
        view_height_ = 0;

        // 
        // 
        gaze_tracker_.setFaceDistance(50);

        gaze_tracker_.setGazeCallback(this);
        gaze_tracker_.setCalibrationCallback(this);

        return true;
    }

    void TrackerManager::setDefaultCameraToDisplayConverter(const seeso::DisplayInfo& display_info) {
        gaze_tracker_.converter() = seeso::makeDefaultCameraToDisplayConverter<float>(
            static_cast<float>(display_info.widthPx), static_cast<float>(display_info.heightPx),
            display_info.widthMm, display_info.heightMm);
    }
    
    void TrackerManager::setViewSize(int width, int height) {
        view_width_ = width;
        view_height_ = height;
    }

    bool TrackerManager::addFrame(std::int64_t timestamp, const cv::Mat& frame) {
        return gaze_tracker_.addFrame(timestamp, frame.data, frame.cols, frame.rows);
    }

    void TrackerManager::startFullWindowCalibration(SeeSoCalibrationPointNum target_num, SeeSoCalibrationAccuracy accuracy) {
        bool expected = false;
        if (!calibrating_.compare_exchange_strong(expected, true))
            return;

        on_calib_start_();

        delayed_calibration_ = std::async(std::launch::async, [=]() {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            const auto window_rect = getWindowRectWithPadding(window_name_.c_str());
            gaze_tracker_.startCalibration(target_num, accuracy,
                window_rect[0], window_rect[1], window_rect[2], window_rect[3]);
            });
    }

    void TrackerManager::setWholeScreenToAttentionRegion(const seeso::DisplayInfo& display_info) {
        gaze_tracker_.setAttentionRegion(0, 0,
            static_cast<float>(display_info.widthPx), static_cast<float>(display_info.heightPx));
    }

    void TrackerManager::calibrateOffset(int screen_center_x, int screen_center_y, int current_gaze_x, int current_gaze_y) {
        // Calculate offset: how far the current gaze is from screen center
        // Example: gaze at (1000, 500), center at (960, 540), offset = (40, -40)
        // After subtracting offset from gaze coordinates, it moves to (960, 540)
        offset_x_ = static_cast<float>(current_gaze_x - screen_center_x);
        offset_y_ = static_cast<float>(current_gaze_y - screen_center_y);
        offset_calibrated_ = true;
        std::cout << "Focus calibration completed: offset (" << offset_x_ << ", " << offset_y_ << ")\n";
        std::cout << "  Raw gaze: (" << current_gaze_x << ", " << current_gaze_y << ")\n";
        std::cout << "  Screen center: (" << screen_center_x << ", " << screen_center_y << ")\n";
        std::cout << "  Expected gaze after correction: (" << (current_gaze_x - static_cast<int>(offset_x_)) 
                  << ", " << (current_gaze_y - static_cast<int>(offset_y_)) << ")\n";
    }
    
    std::pair<int, int> TrackerManager::getCurrentRawGaze() const {
        return { current_raw_gaze_x_, current_raw_gaze_y_ };
    }
    
    void TrackerManager::startMultiPointCalibration(int view_width, int view_height) {
        if (multi_point_calibrating_) {
            std::cout << "Calibration already in progress\n";
            return;
        }
        
        view_width_ = view_width;
        view_height_ = view_height;
        
        // Initialize 5 calibration points: 4 corners + center
        // Add padding to keep calibration points visible (considering circle radius ~30px)
        const int padding = 50;
        
        calibration_points_.clear();
        calibration_points_.resize(5);
        
        // Top-left
        calibration_points_[0].target_x = padding;
        calibration_points_[0].target_y = padding;
        calibration_points_[0].calibrated = false;
        
        // Top-right
        calibration_points_[1].target_x = view_width - padding - 1;
        calibration_points_[1].target_y = padding;
        calibration_points_[1].calibrated = false;
        
        // Bottom-right
        calibration_points_[2].target_x = view_width - padding - 1;
        calibration_points_[2].target_y = view_height - padding - 1;
        calibration_points_[2].calibrated = false;
        
        // Bottom-left
        calibration_points_[3].target_x = padding;
        calibration_points_[3].target_y = view_height - padding - 1;
        calibration_points_[3].calibrated = false;
        
        // Center
        calibration_points_[4].target_x = view_width / 2;
        calibration_points_[4].target_y = view_height / 2;
        calibration_points_[4].calibrated = false;
        
        multi_point_calibrating_ = true;
        multi_point_calibrated_ = false;
        current_calibration_point_ = 0;
        calibration_start_time_ = std::chrono::steady_clock::now();
        last_sample_time_ = calibration_start_time_;
        
        std::cout << "Multi-point calibration started. Look at the top-left corner for 5 seconds...\n";
    }
    
    std::pair<float, float> TrackerManager::getInterpolatedOffset(float gaze_x, float gaze_y) const {
        if (calibration_points_.size() != 5 || !multi_point_calibrated_) {
            return { 0.0f, 0.0f };
        }
        
        // Use bilinear interpolation with 4 corners and center
        // Find the quadrant and interpolate
        
        float center_x = static_cast<float>(view_width_) / 2.0f;
        float center_y = static_cast<float>(view_height_) / 2.0f;
        
        // Determine which quadrant
        bool right = gaze_x > center_x;
        bool bottom = gaze_y > center_y;
        
        // Get corner points
        const auto& tl = calibration_points_[0];  // top-left
        const auto& tr = calibration_points_[1];  // top-right
        const auto& br = calibration_points_[2];  // bottom-right
        const auto& bl = calibration_points_[3];  // bottom-left
        const auto& center = calibration_points_[4];
        
        // Normalize coordinates to [0, 1] range
        float nx = (gaze_x - center_x) / center_x;  // -1 to 1
        float ny = (gaze_y - center_y) / center_y;  // -1 to 1
        
        // Clamp to [-1, 1]
        nx = std::max(-1.0f, std::min(1.0f, nx));
        ny = std::max(-1.0f, std::min(1.0f, ny));
        
        // Convert to [0, 1] for interpolation
        float u = (nx + 1.0f) / 2.0f;
        float v = (ny + 1.0f) / 2.0f;
        
        // Bilinear interpolation
        float offset_x = (1.0f - u) * (1.0f - v) * tl.offset_x +
                        u * (1.0f - v) * tr.offset_x +
                        u * v * br.offset_x +
                        (1.0f - u) * v * bl.offset_x;
        
        float offset_y = (1.0f - u) * (1.0f - v) * tl.offset_y +
                        u * (1.0f - v) * tr.offset_y +
                        u * v * br.offset_y +
                        (1.0f - u) * v * bl.offset_y;
        
        // Blend with center point based on distance
        float dist_from_center = std::sqrt(nx * nx + ny * ny);
        float center_weight = std::max(0.0f, 1.0f - dist_from_center);
        
        offset_x = offset_x * (1.0f - center_weight * 0.3f) + center.offset_x * center_weight * 0.3f;
        offset_y = offset_y * (1.0f - center_weight * 0.3f) + center.offset_y * center_weight * 0.3f;
        
        return { offset_x, offset_y };
    }

} // namespace sample
