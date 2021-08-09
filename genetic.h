#ifndef GENETIC_H
#define GENETIC_H
#include <iostream>
#include <vector>
#include <string>
#include "perceptron.h"

namespace Genetic {
    extern void output_error(std::vector<Perceptron*> &population);

    extern void min_error(std::vector<Perceptron*> &population);

    extern std::vector<Perceptron*> create_population(size_t population_size, Network_Parameters NP);

    extern void do_evolution_step(std::vector<Perceptron*>& population, size_t child_count,
                           ublas::vector<double> input_data, ublas::vector<double> output_data);

    extern void realize_algorithm(Network_Parameters NP, std::vector<std::pair<ublas::vector<double>,ublas::vector<double>>> &dv);
}


#endif // GENETIC_H
