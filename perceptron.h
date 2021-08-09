#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#define learning_rate 0.1

namespace ublas = boost::numeric::ublas;

struct Network_Parameters{
    Network_Parameters(size_t il_size, size_t hl_size, size_t ol_size, size_t hidden_layers){
        il = il_size;
        hl = hl_size;
        ol = ol_size;
        hl_count = hidden_layers;
    }
    size_t il;
    size_t hl;
    size_t ol;
    size_t hl_count;
};

class Layer{
    ublas::vector<double> X;
    ublas::vector<double> B;
    double (*activation)(double);
public:
    Layer(int size, double (*af)(double)){
        X = ublas::vector<double>(size);
        activation = af;
    }
    ublas::vector<double> make_step(ublas::vector<double> x, ublas::matrix<double> W);
    ublas::vector<double> withdrawal_step(ublas::vector<double> x, ublas::matrix<double> W);

    ublas::vector<double> get_data(){ return X; }
    int get_size() const { return X.size();}
    friend std::ostream& operator<< (std::ostream &out, const Layer &l);
};

std::ostream& operator<< (std::ostream &out, const Layer &l);

class Perceptron{
    std::vector<Layer*> layers;
    std::vector<ublas::matrix<double>> W;
    double err_norm;
public:
    Perceptron(Network_Parameters P);
    Perceptron(Perceptron& P){
        layers = P.layers;
        W = P.W;
    }
    void make_iteration(ublas::vector<double> input_data);
    double count_error_norm(ublas::vector<double> data);
    double get_error_norm(){ return err_norm; }

    void mutate();
    void back_propagation(ublas::vector<double> data);

    friend std::ostream& operator<< (std::ostream &out, const Perceptron &p);
};

std::ostream& operator<< (std::ostream &out, const Perceptron &p);

#endif // PERCEPTRON_H
