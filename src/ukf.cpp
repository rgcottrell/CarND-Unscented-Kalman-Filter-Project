#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// To avoid divide by zero errors and numerical instability of small numbers
// close to zero, consider any value less than this to be zero.
static const double EPSILON = 0.001;

// Helper function to normal an angle to the range (-pi, pi).
static double normalize_angle(double angle) {
  while (angle > M_PI) {
    angle -= 2 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2 * M_PI;
  }
  return angle;
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Initial state vector. This will be updated with the first measurement.
  x_ = VectorXd(5);

  // Initial covariance matrix. Start with the identity matrix, which presumes
  // that each component of the state vector is linearly uncorrelated with any
  // other component.
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.7;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  // State dimensions
  n_x_ = 5;
  
  // Augmented state dimensions
  n_aug_ = 7;
  
  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;
  
  // Predicted sigma points matrix. This will be populated in the predict
  // step and later used by the update step.
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  // Weights of sigma points.
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // If this is the first measurement, complete state initialization.
  // Process the first measurement even if future sensor measurements
  // should be ignored.
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Radar provides positional spherical coordinates which will need to
      // be converted to cartesian coordinates for the state vector.
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double px = rho * cos(phi);
      double py = rho * sin(phi);
      x_ << px, py, 0, 0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Lidar only provides positional measurements. Default the rest of
      // the state parameters to zero.
      double px = meas_package.raw_measurements_(0);
      double py = meas_package.raw_measurements_(1);
      x_ << px, py, 0, 0, 0;
    }
    
    // Set the initial timestamp.
    time_us_ = meas_package.timestamp_;
    
    // Done initializing. No need to predict or update.
    is_initialized_ = true;
    return;
  }
  
  // Check to see if the sensor type should be processed or skipped.
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) {
    return;
  }
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_) {
    return;
  }
  
  // Compute the time elapsed in seconds between the current and previous
  // measurements.
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;

  // Perform the prediction step to update internal state since the last
  // measurement.
  Prediction(delta_t);
  
  // Use the sensor type to perform the update step. Update the state and
  // covariance matrices.
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
  
  // Save the baseline timestamp for the next measurement.
  time_us_ = meas_package.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Create augmented state vector.
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_ + 0) = 0;
  x_aug(n_x_ + 1) = 0;
  
  // Create augmented covariance matrix.
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_ + 0, n_x_ + 0) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
  
  // Create square root matrix.
  MatrixXd L = P_aug.llt().matrixL();
  
  // Create the augmented sigma point matrix.
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  // Apply process model to transform sigma points to predicted values at
  // the new timestamp.
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);
    
    // Avoid division by zero errors by assuming linear, rather than curved,
    // movement for small values of yaw.
    double px_p;
    double py_p;
    if (fabs(yawd) >= EPSILON) {
      px_p = px + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = py + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = px + v * delta_t * cos(yaw);
      py_p = py + v * delta_t * sin(yaw);
    }
    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;
    
    // Add noise.
    px_p += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p += nu_a * delta_t;
    yaw_p += 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p += nu_yawdd * delta_t;

    // Store predicted sigma points for use in the update step.
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  // Calculate new predicted state mean.
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // Calculate new predicted state covariance matrix.
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = normalize_angle(x_diff(3));
    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Number of measurment dimensions (px, py).
  int n_z = 2;
  
  // Transform sigma points into measurement space.
  MatrixXd Zsig(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    Zsig(0, i) = px;
    Zsig(1, i) = py;
  }

  // Mean predicted measurement.
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred += weights_(i) * Zsig.col(i);
  }
  
  // Measurement covariance matrix.
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }
 
  // Add measurement noise covariance matrix.
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;
  S += R;

  // Cross correlation matrix.
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++ i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  
  // Kalman gain matrix.
  MatrixXd K = Tc * S.inverse();
  
  // Residual between measured and predicted.
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  
  // Finally, update the state mean and process covariance matrix.
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Number of measurment dimensions (rho, phi, rho_dot).
  int n_z = 3;
  
  // Transform sigma points into measurement space.
  MatrixXd Zsig(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double vx = v * cos(yaw);
    double vy = v * sin(yaw);
    Zsig(0, i) = sqrt(px * px + py * py);
    Zsig(1, i) = atan2(py, px);
    Zsig(2, i) = (px * vx + py * vy) / sqrt(px * px + py * py);
  }

  // Mean predicted measurement.
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred += weights_(i) * Zsig.col(i);
  }
  
  // Measurement covariance matrix.
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = normalize_angle(z_diff(1));
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  
  // Add measurement noise covariance matrix.
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_ * std_radr_, 0, 0,
       0, std_radphi_ * std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;
  S += R;
  
  // Cross correlation matrix.
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++ i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = normalize_angle(z_diff(1));
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = normalize_angle(x_diff(3));
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  
  // Kalman gain matrix.
  MatrixXd K = Tc * S.inverse();
  
  // Residual between measured and predicted.
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  z_diff(1) = normalize_angle(z_diff(1));
  
  // Finally, update the state mean and process covariance matrix.
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
