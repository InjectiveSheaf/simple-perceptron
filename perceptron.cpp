#include "perceptron.h"
#include "mathfunctions.h"

std::ostream& operator<< (std::ostream &out, const Layer &l){
    int size = l.get_size();
    if(size > 15) return out << "size = " << size;
    for(int i=0; i < size-1; i++) out << "O-";
    out << "O";
    return out;
}

ublas::vector<double> Layer::make_step(ublas::vector<double> x, ublas::matrix<double> W){
    ublas::vector<double> tmp = ublas::prod(W,x);
    std::transform(tmp.begin(), tmp.end(), tmp.begin(), activation);
    X = tmp;
    return X;
}

ublas::vector<double> Layer::withdrawal_step(ublas::vector<double> x, ublas::matrix<double> W){
    std::cout << std::endl << W << "*" << x << "+" << B << std::endl;
    ublas::vector<double> tmp = ublas::prod(W,x);
    std::transform(tmp.begin(), tmp.end(), tmp.begin(), activation);
    X = tmp;
    std::cout << "Result: " << X << std::endl;
    return X;
}

std::ostream& operator<< (std::ostream &out, const Perceptron &p){
    size_t i;
    out << "This perceptron has " << p.layers.size() << " layers and " << p.W.size() << " matrices." << std::endl;
    for(i=0; i < p.layers.size()-1; i++){
        out << std::endl << "Layer " << i << ": " << *p.layers[i] << std::endl;
        out << "Matrix of weights: " << std::endl << p.W[i];
        out << "Layer " << i+1 << ": " << *p.layers[i+1] << std::endl;
    }
    return out;
}

Perceptron::Perceptron(Network_Parameters P){
    Layer* input_layer = new Layer(P.il, AF::sigm);
    layers.push_back(input_layer);
    for(size_t i = 0; i < P.hl_count; i++){
        Layer* hidden_layer = new Layer(P.hl, AF::sigm);
        layers.push_back(hidden_layer);
    }
    Layer* output_layer = new Layer(P.ol, AF::sigm);
    layers.push_back(output_layer);

    W.push_back(CreateRandomMatrix(P.hl,P.il));
    for(size_t i = 0; i < P.hl_count-1; i++) W.push_back(CreateRandomMatrix(P.hl,P.hl));
    W.push_back(CreateRandomMatrix(P.ol,P.hl));
}

void Perceptron::make_iteration(ublas::vector<double> input_data){
    layers[1]->make_step(input_data, W[0]);
    for(size_t i = 2; i < layers.size(); ++i){
        layers[i]->make_step(layers[i-1]->get_data(), W[i-1]);
    }
}

double Perceptron::count_error_norm(ublas::vector<double> data){
    ublas::vector<double> x = layers.back()->get_data();
    //std::cout << data << " - " << x << std::endl;
    ublas::vector<double> delta = data - x;
    //std::cout << delta << std::endl;
    err_norm = ublas::norm_2(delta);
    return err_norm;
}

void Perceptron::mutate(){
    for(size_t i = 0; i < W.size(); i++){
        W[i] += CreateRandomMatrix(W[i].size1(),W[i].size2());
    }
}

void Perceptron::back_propagation(ublas::vector<double> data){
    for(size_t i = layers.size()-1; i != 0; i--){
        ublas::vector<double> x = layers[i]->get_data();
        ublas::vector<double> delta = data - x;

        std::cout << "Err = " << delta << std::endl;

        ublas::matrix<double> dW( W[i-1].size1(), W[i-1].size2() );
        std::cout << dW << std::endl;
        for (size_t k = 0; k != W[i-1].size2(); ++k){
            for (size_t j = 0; j != W[i-1].size1(); ++j){
                dW(j,k) = learning_rate * delta(k) * x(j);
                std::cout << j << k << dW(j,k) << std::endl;
            }
        }
        std::cout << W[i-1] << " - " << dW << std::endl;
        W[i-1] = W[i-1] - dW;
        std::cout << W[i-1] << std::endl;
    }
}

