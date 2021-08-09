#include "genetic.h"

namespace Genetic {
    void output_error(std::vector<Perceptron *> &population){
        std::cout << "Population: " << std::endl;
        for(size_t p = 0; p < population.size(); p++)
            std::cout << "error["<< p << "] = " << population[p]->get_error_norm() << "; ";
        std::cout << std::endl;
    }

    void min_error(std::vector<Perceptron*> &population){
        std::cout << "Error: " << population[0]->get_error_norm() * 10 << "%" << std::endl;
    }

    std::vector<Perceptron*> create_population(size_t population_size, Network_Parameters NP){
        std::vector<Perceptron*> population;
        for(size_t i = 0; i < population_size; i++){
            Perceptron* P = new Perceptron(NP);
            population.push_back(P);
        }
        return population;
    }

    void do_evolution_step(std::vector<Perceptron*>& population, size_t child_count,
                           ublas::vector<double> input_data, ublas::vector<double> output_data){
        size_t p_size = population.size();
        for(size_t i = 0; i < p_size; i++){ // размножаем перцептроны
            for(size_t j = 0; j < child_count; j++){ // проводим шаг эволюции для размножившейся популяции
                Perceptron* P = new Perceptron(*population[i]);
                P->mutate();
                P->make_iteration(input_data);
                P->count_error_norm(output_data);
                population.push_back(P);
            }
        }
    }

    /* отправляем в качестве данных указатель на data_vector, определенный как
     * data_vector = вектор< пара<вектор входных, вектор выходных> > численных данных
     */
    void realize_algorithm(Network_Parameters NP, std::vector<std::pair<ublas::vector<double>,ublas::vector<double>>> &dv){
        using namespace std;
        ublas::vector<double> input_data(NP.il);
        ublas::vector<double> output_data(NP.ol);
        int iteration = 0;

        // создаём первичную популяцию
        size_t p_size = 10, child_count = 10;
        vector<Perceptron*> population = create_population(p_size, NP);

        for(const auto &it : dv){ /* итерируемся по всему data_vector-у */
            std::cout << iteration << std::endl;
            iteration++;

            input_data = it.first; // берем первый массив из пары
            output_data = it.second; // берем второй массив из пары
            // проводим шаг генетического алгоритма (обучения)
            for(size_t i = 0; i < population.size(); i++){
                population[i]->make_iteration(input_data);
                population[i]->count_error_norm(output_data);
            }
            do_evolution_step(population, child_count, input_data, output_data);
            std::sort(population.begin(),population.end(),[](Perceptron* A, Perceptron* B){
                return A->get_error_norm() < B->get_error_norm();
            });
            population.erase(population.begin()+p_size, population.end());
            // смотрим ошибку
            min_error(population);
        }
    } 
}
/*
 во вводе данных до этого было:
 ublas::vector<double> input_data(NP.il);
 ublas::vector<double> output_data(NP.ol);
 while(true){
*for(size_t i = 0; i < NP.il; i++)  std::cin >> input_data(i);
*for(size_t i = 0; i < NP.ol; i++)  std::cin >> output_data(i);
*... */



