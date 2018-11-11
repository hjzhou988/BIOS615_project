//
//  main.cpp
//  615Project
//
//  Created by Huajun Zhou on 11/11/18.
//  Copyright © 2018 Huajun Zhou. All rights reserved.
//

#include <iostream>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <Eigen/Dense>
#include <boost/random/uniform_real.hpp>


using namespace std;
using namespace boost;
using namespace Eigen;

int main(int argc, const char * argv[]) {
    
    boost::mt19937 rng;
    rng.seed(399);
    int N_sample=1000;
    
    VectorXd X(2);
    VectorXd M(2);
    VectorXd Z(2);
    MatrixXd Sigma(2,2);
    double rho=0.5;
    Sigma << 1,rho,
    rho,1;
    M << 10,10;
    
    LLT<MatrixXd> lltOfSigma(Sigma);
    MatrixXd L=lltOfSigma.matrixL();
    
    
    float sigma=0.5;
    boost::random::normal_distribution<> z_normal(10,sigma);
    
    
    uniform_real<> uni(0,1);
    double beta_0=uni(rng), beta_1=uni(rng), beta_2=1-beta_0-beta_1;
    
    
    
    // generate a 1000 x 4 matrix;
    Matrix<double, 1000, 4> Network;
    for (int i=0; i<1000; i++)
    {
        //generate X;
        boost::random::normal_distribution<> std_normal(0,1.0);
        for (int j=0;j<2; ++j)
        {
            Z(j)=std_normal(rng);
        }
        X = L * Z + M;
        Network(i,0)= X(0);
        Network(i,1)=X(1);
        
        //Generate z;
        double z=z_normal(rng);
        Network(i,2)=z;
        
        //Gererate y;
        double y= beta_0 + beta_1 * X(0) + beta_2 * X(1);
        Network(i,3)=y;
    }
    
    cout<< Network;
    
    
    return 0;
}

