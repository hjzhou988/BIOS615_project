/*
    main.cpp
    615Project

    Created by Huajun Zhou on 11/11/18.
    Copyright © 2018 Huajun Zhou. All rights reserved.
    
    How to install gsl library:
    1. Go to https://www.gnu.org/software/gsl/
    2. Download and unzip the file
    3. move the gsl-2.5 folder to /include folder
    4. read the INSTALL file, and follow the make install steps
    5. Done
    
    How to compile:
    g++ -I/usr/local/include -L/usr/local/lib -lgsl main.cpp -o main
    
    How to run:
    ./main
*/



#include <iostream>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <Eigen/Dense>
#include <boost/random/uniform_real.hpp>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>



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

    // generate a 1000 x 4 Population Data Matrix;
    Matrix<double, Dynamic, Dynamic> Population_data;
    Population_data.resize(N_sample,4);
    for (int i=0; i<N_sample; i++)
    {
        //generate X;
        boost::random::normal_distribution<> std_normal(0,1.0);
        for (int j=0;j<2; ++j)
        {
            Z(j)=std_normal(rng);
        }
        X = L * Z + M;
        Population_data(i,1)= X(0);
        Population_data(i,2)=X(1);

        //Generate z;
        double z=z_normal(rng);
        Population_data(i,3)=z;

        //Gererate y;
        double y= beta_0 + beta_1 * X(0) + beta_2 * X(1);
        Population_data(i,0)=y;
    }

    // cout<< Population_data<<endl;

    //generate a 1000*1000 Population Network Matrix
    Matrix<double, Dynamic, Dynamic> Population_network;
    Population_network.resize(N_sample,N_sample);
    for (int i=0; i<N_sample; i++)
    {
        //row ~ Bin(n=N_sample, p=5/N_sample);
        boost::random::binomial_distribution<> bin_dist(N_sample,(double)5/N_sample);
        double num_friend = bin_dist(rng);
        //cout<<"num_friend "<<num_friend<<endl;

        double p[N_sample];
        unsigned int n[N_sample];
        for(int count=0; count<N_sample;count++){
          p[count]=(double)1/N_sample;
        }
        //void gsl_ran_multinomial (const gsl_rng * r, size_t K=1000, unsigned int N=num_friend, const double p[], unsigned int n[])
        const gsl_rng_type * T;
        gsl_rng * r;
        gsl_rng_env_setup();
        T = gsl_rng_default;
        r = gsl_rng_alloc (T);
        gsl_rng_set(r, (unsigned long int) i*314159265 );
        const size_t K = N_sample;
        gsl_ran_multinomial(r, K, num_friend, p, n);
        for(int j=0;j<N_sample;j++){
          if(i!=j){
            Population_network(i,j)=n[j];
          }else{
            Population_network(i,j)=0;
          }

        }


    }
    // cout<<Population_network<<endl; //Original population_network matrix
    Population_network = Population_network.triangularView<Upper>();

    Matrix<double, Dynamic, Dynamic> network_temp;
    network_temp = Population_network;
    network_temp.transposeInPlace();

    // Finalized symmetric network matrix with diagnol of zeros
    Population_network += network_temp;
    // cout<<Population_network<<endl;  //Final Population_network matrix

    return 0;
}
