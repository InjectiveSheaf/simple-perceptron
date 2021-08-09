#include "perceptron.h"
#include "genetic.h"

#include <experimental/filesystem>
#include <QImage>
#include <QString>
#include <algorithm>
#include <random>

namespace fs = std::experimental::filesystem;

int main()
{
    std::string dir = "/home/lucretia/Рабочий стол/ml_datasets/Cutout Files";
    fs::path star = dir + "/star" ;
    fs::path galaxy = dir + "/galaxy";
    std::vector<std::pair<std::string,std::string>> data_paths;
    for(const auto &entry : fs::recursive_directory_iterator(dir)){ // спарсили картиночки и дали им типы
        std::string temp;
        if(entry.path().parent_path() == star) temp = "star";
        if(entry.path().parent_path() == galaxy) temp = "galaxy";
        if(!temp.empty()) {
        data_paths.push_back(std::make_pair(entry.path(),temp));
        }
    }

    std::vector<std::pair<ublas::vector<double>,ublas::vector<double>>> dv;
    for(size_t i = 0; i < data_paths.size(); i++){
        QImage image = QImage(QString::fromStdString(data_paths[i].first));
        int height = image.height();
        int width = image.width();

        ublas::vector<double> type_vector(1); // вектор типа, по факту состоящий из одного элемента
        if(data_paths.at(i).second == "star") type_vector[0] = 0;
        if(data_paths.at(i).second == "galaxy") type_vector[0] = 1;

        ublas::vector<double> image_vector(width*height);
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                int n = width*y + x;
                image_vector[n] = qGray(image.pixel(x, y));
            }
        }
        image_vector = image_vector/ublas::norm_2(image_vector); // нормируем вектор, полученный из изображения
        dv.push_back(make_pair(image_vector,type_vector));
    }
    auto rng = std::default_random_engine {};
    std::shuffle(dv.begin(), dv.end(), rng);
    std::cout << "Data has been loaded, number of samples: " << dv.size() << std::endl;

    time_t t = time(nullptr);
    srand(t);
    size_t il_size = 64*64, hl_size = 10, ol_size = 1, hl = 10;
    Network_Parameters NP(il_size, hl_size, ol_size, hl);
    Genetic::realize_algorithm(NP, dv);
}
