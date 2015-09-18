#include "core.hpp"
#include "its.hpp"
#include "features.hpp"
#include "pythonBindings.hpp"

int main(int argc, char** argv) {
	std::cout << "Rather use the application as library." << std::endl;

	// example of using the basic template algorithm from C++
	// (boost::shared_ptr are needed due to Python bindings)

	its::AlgorithmTemplate a;
	a.Attach(boost::make_shared<its::ImagePhenotype>(256)); // via attach
	a.Features.push_back(boost::make_shared<its::Std>()); // or implicitly
	a.Attach(boost::make_shared<its::Sum>());

	for(int i = 0; i < 10; ++i) {
		a.RunOneGeneration();
		std::cout << a.AggregationFunction->Fitness << std::endl; 
	}

	return 0;
}
