#ifndef MATHFUNCTIONS_H
#define MATHFUNCTIONS_H

#define learning_rate 0.1

#include <iostream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/operators.hpp>

namespace ublas = boost::numeric::ublas;

namespace AF{
    double heaviside(double v){
        if (v >= 0) return 1;
        else return 0;
    }
    double pcw_linear(double v){
        if (v >= 0.5) return 1;
        else if (v <= -0.5) return 0;
        else return std::abs(v);
    }
    double sigm(double v){
        double k = 1;
        return 1/(1+std::exp(-k*v));
    }
    double sigm_deriv(double v){
        return sigm(v)*(1-sigm(v));
    }
}

const ublas::matrix<double> CreateRandomMatrix(const std::size_t n_rows, const std::size_t n_cols){
    ublas::matrix<double> m(n_rows,n_cols);
    for (std::size_t row=0; row!=n_rows; ++row){
        for (std::size_t col=0; col!=n_cols; ++col){
          m(row,col) = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX) - 0.5;
        }
    }
    return m;
}

template<typename T>
std::ostream& operator<< (std::ostream &out, const ublas::matrix<T> &m){
    for(unsigned i=0;i<m.size1();++i){
        for (unsigned j=0;j<m.size2();++j){
            out<<m(i,j)<<" | ";
        }
        out << std::endl;
    }
    return out;
}

#endif // MATHFUNCTIONS_H
