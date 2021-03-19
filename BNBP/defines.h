#include <vector>
#include <Eigen/Dense>

//#define __USE_DOUBLE__

#ifdef __USE_DOUBLE__
typedef double Float;
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;
#else
typedef float Float;
typedef Eigen::MatrixXf Mat;
typedef Eigen::VectorXf Vec;
#endif //__USE_DOUBLE__
