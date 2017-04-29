#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
    
  // Check the validity of the inputs.
  if (estimations.size() != ground_truth.size()
          || estimations.size() == 0) {
    cout << "CalculateRMSE: Invalid estimation or ground_truth data" << endl;
    return rmse;
  }
    
  // Accumulate the squared residuals.
  for (int i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }
    
  // Calculate the mean.
  rmse = rmse / estimations.size();
    
  // Calculate the square root.
  rmse = rmse.array().sqrt();
    
  // Return the final RMSE.
  return rmse;
}
