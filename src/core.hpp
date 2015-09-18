#ifndef CORE_HPP
#define	CORE_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <set>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <fstream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <opencv2/opencv.hpp>

#include "MultiNEAT/lib/Population.h"

#include "utilities.hpp"

namespace its {

	// forward declaration of AlgorithmTemplate 
	struct AlgorithmTemplate;

	// Component that can be used by an algorithm.
	struct Component {
		// Attaches the component to the given algorithm.
		// Note: This component is given as shared_ptr in parameters. This solves problem when it was created by Python. 
		virtual void Attach(AlgorithmTemplate& algorithm, boost::shared_ptr<Component> c) = 0;
	};
	typedef boost::shared_ptr<Component> ComponentPtr;



	/// DEFINITION OF COMPONENTS

	// Phenotype function to create an image from an individual.
	struct PhenotypeFunction : virtual public Component {
		// Image representation of the last given individual.
		cv::Mat Image;
		// Neural network of the last given individual.
		NEAT::NeuralNetwork NeuralNetwork;

		cv::Mat GetImage() { return Image; } // for python binding

		// Creates an image representation of the given individual. 
		// The result is stored in the variable Image.
		virtual void Create(NEAT::Genome& individual) = 0;
	};
	typedef boost::shared_ptr<PhenotypeFunction> PhenotypeFunctionPtr;

	// Feature function to evaluate an individual.
	struct FeatureFunction : virtual public Component {
		// Evaluates the given individual.
		virtual double Evaluate(const cv::Mat& image) = 0;
	};
	typedef boost::shared_ptr<FeatureFunction> FeatureFunctionPtr;

	// Aggregation function to aggregate results from all features.
	struct AggregationFunction : virtual public Component  {
		// Fitness value of the last given individual.
		// Note: non-scaled value = value from the original feature space.
		double Fitness;
		// Scaled fitness value of the last given individual.
		double ScaledFitness;

		// Aggregates values to a single fitness value.
		// The results is stored in Fitness and ScaledFitness.
		virtual void Aggregate(const std::vector<double>& values) = 0;
	};
	typedef boost::shared_ptr<AggregationFunction> AggregationFunctionPtr;

	// Action called before an evaluation of population.
	struct BeforePopulationEvaluationAction : virtual public Component  {
		// Is called before an evaluation of population.
		virtual void BeforePopulationEvaluation(AlgorithmTemplate& algorithm) = 0;
	};
	typedef boost::shared_ptr<BeforePopulationEvaluationAction> BeforePopulationEvaluationActionPtr;

	// Action called after an evaluation of population.
	struct AfterPopulationEvaluationAction : virtual public Component  {
		// Is called after an evaluation of population.
		virtual void AfterPopulationEvaluation(AlgorithmTemplate& algorithm) = 0;
	};
	typedef boost::shared_ptr<AfterPopulationEvaluationAction> AfterPopulationEvaluationActionPtr;

	// Action called before an evaluation of individual.
	struct BeforeIndividualEvaluationAction : virtual public Component  {
		// Is called before an evaluation of individual.
		virtual void BeforeIndividualEvaluation(AlgorithmTemplate& algorithm) = 0;
	};
	typedef boost::shared_ptr<BeforeIndividualEvaluationAction> BeforeIndividualEvaluationActionPtr;

	// Action called after an evaluation of individual.
	struct AfterIndividualEvaluationAction : virtual public Component  {
		// Is called after an evaluation of individual.
		virtual void AfterIndividualEvaluation(AlgorithmTemplate& algorithm) = 0;
	};
	typedef boost::shared_ptr<AfterIndividualEvaluationAction> AfterIndividualEvaluationActionPtr;

	// Action called only once after an initialization of an algorithm.
	struct InitAction : virtual public Component  {
		// Is called after an initialization of an algorithm.
		virtual void Init(AlgorithmTemplate& algorithm) = 0;
	};
	typedef boost::shared_ptr<InitAction> InitActionPtr;



	/// DEFINITION OF GENERAL ALGORITHM

	typedef boost::shared_ptr<NEAT::Population> PopulationPtr; // shared_ptr of Neat population

	// General algorithm for NEAT implemented as a component system.
	// It provides only the basic functionality and other behaviour is achieved by attached components.
	class AlgorithmTemplate {
	protected:
		bool initialized = false; // has the algorithm been initialized?

	public:
		// Population of NEAT.
		PopulationPtr Population = nullptr;
		// NEAT parameters to initialize the initial population.
		NEAT::Parameters Parameters;
		// Output function for an individual's neural network. 
		NEAT::ActivationFunction outputActivationFn = NEAT::SIGNED_GAUSS;

		// Phenotype function to create an image from an individual.
		PhenotypeFunctionPtr PhenotypeFunction = nullptr;
		// Feature functions to evaluate an individual.
		std::vector<FeatureFunctionPtr> Features;
		// Aggregation function to aggregate results from all features.
		AggregationFunctionPtr AggregationFunction = nullptr;

		// Actions called before an evaluation of the population.
		std::vector<BeforePopulationEvaluationActionPtr> BeforePopulationEvaluationActions;
		// Actions called after an evaluation of the population.
		std::vector<AfterPopulationEvaluationActionPtr> AfterPopulationEvaluationActions;
		// Actions called before an evaluation of an individual.
		std::vector<BeforeIndividualEvaluationActionPtr> BeforeIndividualEvaluationActions;
		// Actions called after an evaluation of an individual.
		std::vector<AfterIndividualEvaluationActionPtr> AfterIndividualEvaluationActions;
		// Actions called after the initialization.
		std::vector<InitActionPtr> InitActions;

		// Number of the current generation.
		int Generation = 1;

		AlgorithmTemplate() {
			// DEFAULT PARAMETERS for NEAT

			// Note: comments are taken from MultiNEAT library

			////////////////////
			// Basic parameters
			////////////////////

			// Size of population
			Parameters.PopulationSize = 150;

			// Minimum number of species
			Parameters.MinSpecies = 10;

			// Maximum number of species
			Parameters.MaxSpecies = 15;


			////////////////////////////////
			// GA Parameters
			////////////////////////////////

			// Age treshold, meaning if a species is below it, it is considered young
			Parameters.YoungAgeTreshold = 15;

			// Fitness boost multiplier for young species (1.0 means no boost)
			// Make sure it is >= 1.0 to avoid confusion
			Parameters.YoungAgeFitnessBoost = 1.1;

			// Number of generations without improvement (stagnation) allowed for a species
			Parameters.SpeciesMaxStagnation = 15;

			// Age threshold, meaning if a species is above it, it is considered old
			Parameters.OldAgeTreshold = 35;

			// Percent of best individuals that are allowed to reproduce. 1.0 = 100%
			Parameters.SurvivalRate = 0.2;

			// Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100%
			// If asexual reprodiction is chosen, the baby will be mutated 100%
			Parameters.CrossoverRate = 0.75;

			// If a baby results from sexual reproduction, this probability determines if mutation will
			// be performed after crossover. 1.0 = 100% (always mutate after crossover)
			Parameters.OverallMutationRate = 0.25;

			// Probability for a baby to result from inter-species mating.
			Parameters.InterspeciesCrossoverRate = 0.01;

			// Probability for a baby to result from Multipoint Crossover when mating. 1.0 = 100%
			// The default is the Average mating.
			Parameters.MultipointCrossoverRate = 0.6;


			///////////////////////////////////
			// Structural Mutation parameters
			///////////////////////////////////

			// Probability for a baby to be mutated with the Add-Neuron mutation.
			Parameters.MutateAddNeuronProb = 0.05;

			// Probability for a baby to be mutated with the Add-Link mutation
			Parameters.MutateAddLinkProb = 0.04;

			// Probability for a baby to be mutated with the Remove-Link mutation
			Parameters.MutateRemLinkProb = 0.04;

			// Probability for a baby that a simple neuron will be replaced with a link
			Parameters.MutateRemSimpleNeuronProb = 0.001;

			// Maximum number of tries to find 2 neurons to add/remove a link
			Parameters.LinkTries = 32;

			// Probability that a link mutation will be made recurrent
			Parameters.RecurrentProb = 0.0;

			// Probability that a recurrent link mutation will be looped
			Parameters.RecurrentLoopProb = 0.0;


			///////////////////////////////////
			// Parameter Mutation parameters
			///////////////////////////////////

			// Probability for a baby's weights to be mutated
			Parameters.MutateWeightsProb = 0.9;

			// Probability for a severe (shaking) weight mutation
			Parameters.MutateWeightsSevereProb = 0.5;

			// Probability for a particular gene's weight to be mutated. 1.0 = 100%
			Parameters.WeightMutationRate = 0.75;

			// Maximum perturbation for a weight mutation
			Parameters.WeightMutationMaxPower = 1.0;

			// Maximum magnitude of a replaced weight
			Parameters.WeightReplacementMaxPower = 2.0;

			// Maximum absolute magnitude of a weight
			Parameters.MaxWeight = 8.0;

			// Probability for a baby that an activation function type will be changed for a single neuron
			// considered a structural mutation because of the large impact on fitness
			Parameters.MutateNeuronActivationTypeProb = 0.02;

			// Probabilities for a particular activation function appearance
			Parameters.ActivationFunction_SignedSigmoid_Prob = 1.0;
			Parameters.ActivationFunction_UnsignedSigmoid_Prob = 0.0;
			Parameters.ActivationFunction_Tanh_Prob = 0.0;
			Parameters.ActivationFunction_TanhCubic_Prob = 0.0;
			Parameters.ActivationFunction_SignedStep_Prob = 0.0;
			Parameters.ActivationFunction_UnsignedStep_Prob = 0.0;
			Parameters.ActivationFunction_SignedGauss_Prob = 1.0;
			Parameters.ActivationFunction_UnsignedGauss_Prob = 0.0;
			Parameters.ActivationFunction_Abs_Prob = 0.0;
			Parameters.ActivationFunction_SignedSine_Prob = 1.0;
			Parameters.ActivationFunction_UnsignedSine_Prob = 0.0;
			Parameters.ActivationFunction_Linear_Prob = 1.0;


			/////////////////////////////////////
			// Speciation parameters
			/////////////////////////////////////

			// Percent of disjoint genes importance
			Parameters.DisjointCoeff = 2.0;

			// Percent of excess genes importance
			Parameters.ExcessCoeff = 2.0;

			// Average weight difference importance
			Parameters.WeightDiffCoeff = 1.0;

			// Activation function type difference importance
			Parameters.ActivationFunctionDiffCoeff = 1.0;

			// Compatibility treshold
			Parameters.CompatTreshold = 6.0;

			// Minumal value of the compatibility treshold
			Parameters.MinCompatTreshold = 0.2;

			// Modifier per generation for keeping the species stable
			Parameters.CompatTresholdModifier = 0.3;
		}

		// Attaches the given component to the algorithm.
		void Attach(ComponentPtr c) {
			c->Attach(*this, c);
		}

		// Initializes the algorithm.
		virtual bool Init() {
			if(initialized) {
				std::cerr << "Already initialized" << std::endl;
				return false;
			}

			// checking of needed components
			if(PhenotypeFunction == nullptr) {
				std::cerr << "No phenotype function defined" << std::endl;
				return false;
			}
			if(Features.size() == 0) {
				std::cerr << "No features defined" << std::endl;
				return false;
			}
			if(AggregationFunction == nullptr) {
				std::cerr << "No aggregation function defined" << std::endl;
				return false;
			}

			// create the initial population, if needed
			if(Population == nullptr) ResetNEAT();

			// call actions for this situation
			for(int i = 0; i < InitActions.size(); ++i) InitActions[i]->Init(*this);

			initialized = true;

			return true;
		}

		void ResetNEAT() {
			Population = boost::make_shared<NEAT::Population>(NEAT::Genome(0, 4, 0, 1, false, outputActivationFn, NEAT::SIGNED_GAUSS, 0, Parameters), Parameters, true, 1.0);
		}

		// Evaluates the given individual.
		virtual void EvaluateIndividual(NEAT::Genome& individual) {
			for(int i = 0; i < BeforeIndividualEvaluationActions.size(); ++i) BeforeIndividualEvaluationActions[i]->BeforeIndividualEvaluation(*this);

			// create phenotype
			PhenotypeFunction->Create(individual);

			// evaluate all features
			individual.fitness.clear();
			for(int featureIndex = 0; featureIndex < Features.size(); ++featureIndex) {
				const double featureValue = Features[featureIndex]->Evaluate(PhenotypeFunction->Image);
				individual.fitness.push_back(featureValue);
			}

			// aggregate all feature values into a single fitness value
			AggregationFunction->Aggregate(individual.fitness);

			// set individual's fitness value
			individual.SetFitness(AggregationFunction->Fitness, AggregationFunction->ScaledFitness);
			individual.SetEvaluated();

			for(int i = 0; i < AfterIndividualEvaluationActions.size(); ++i) AfterIndividualEvaluationActions[i]->AfterIndividualEvaluation(*this);
		}

		// Evaluates the given population.
		void EvaluatePopulation(NEAT::Population& population) {
			for(int i = 0; i < BeforePopulationEvaluationActions.size(); ++i) BeforePopulationEvaluationActions[i]->BeforePopulationEvaluation(*this);

			// evaluate all individuals in the population
			for(int speciesIndex = 0; speciesIndex < population.m_Species.size(); ++speciesIndex) {
				for(int individualIndex = 0; individualIndex < population.m_Species[speciesIndex].m_Individuals.size(); ++individualIndex) {
					EvaluateIndividual(population.m_Species[speciesIndex].m_Individuals[individualIndex]);
				}
			}

			for(int i = 0; i < AfterPopulationEvaluationActions.size(); ++i) AfterPopulationEvaluationActions[i]->AfterPopulationEvaluation(*this);
		}

		// Runs one generation of the algorithm.
		void RunOneGeneration() {
			// initialize the algorithm, if needed
			if(!initialized) {
				Init();
				EvaluatePopulation(*Population); // extra evaluation to set individuals's fitness values
			}

			// one iteration of NEAT
			Population->Epoch();

			// evaluate population
			EvaluatePopulation(*Population);

			// increase the number of generations
			++Generation;
		}

		// Runs the algorithm until the given number of stagnations.
		void RunUntilStagnation(const int stagnationLimit) {
			do {
				RunOneGeneration();
			} while(Population->GetStagnation() <= stagnationLimit);
		}

		// Runs the algorithm for the fixed number of iterations.
		void RunGenerations(const int iterations) {
			for(int i = 0; i < iterations; ++i) {
				RunOneGeneration();
			}
		}

		// Resets information about a stagnation.
		void ResetStagnation() {
			Population->m_GensSinceBestFitnessLastChanged = 0;
			Population->m_BestFitnessEver = 0;
		}
	};



	/// HELPERS FOR AUTOMATIC ATTACHMENT OF COMPONENTS

	// Its specialization contains static method to attach a component of the particular type to the algorithm. 
	template<typename T>
	struct AutomaticAttach;

	// Provides recursive calls for every type to call a static function on AutomaticAttach class.
	template<typename ...Params>
	struct CallAttach;

	template<typename T, typename ...Params>
	struct CallAttach<T, Params...> {
		static void Call(AlgorithmTemplate& algorithm, ComponentPtr c) {
			// attach the component to the algorithm
			AutomaticAttach<T>::Attach(algorithm, boost::dynamic_pointer_cast<T>(c));
			// resursive call for the rest
			CallAttach<Params...>::Call(algorithm, c);
		}
	};

	template<>
	struct CallAttach<> {
		static void Call(AlgorithmTemplate& algorithm, ComponentPtr c) {} // end of recursion
	};

	// Helper class for using components and their automatic attachment.
	// (via variadic templates)
	template<typename ...Params>
	struct ComponentHelper : public Params... {
		// Attaches the component to the algorithm.
		virtual void Attach(AlgorithmTemplate& algorithm, ComponentPtr c) {
			CallAttach<Params...>::Call(algorithm, c);
		}
	};


	
	/// PHENOTYPE FUNCTIONS

	class ImagePhenotype : public ComponentHelper<PhenotypeFunction> {
	protected:
		const int maxValue; // max value for a coordinate at the image
		const double halfMaxValue; // max value for a coordinate at the half of the image
		const double maxDistanceFromCenter; // maximal distance from the center of the image
		std::vector<double> input; // input for the neural network

		inline double TransformInput(const double x, const double max) const {
			return 2.0 * x / max - 1.0;
		}

		inline double TransformOutput(const double x) const {
			// const double value = (x + 1.0) * 255.0 / 2.0; // scaling function
			const double value = std::abs(x) * 255.0; // from black to white in both directions
			return std::max(0.0, std::min(255.0, value));
		}

	public:

		// Number of neural network activations. Value less or equal to 0 uses depth of the network.
		int NetworkActivations = -1;
		// Number of neural network layers of the last processed individual.
		int NetworkDepth;

		ImagePhenotype(const int size) : maxValue(size - 1), halfMaxValue((size - 1.0) / 2.0), maxDistanceFromCenter((size - 1.0) / std::sqrt(2.0)), input(4) {
			Image = cv::Mat(size, size, CV_8UC1, cv::Scalar::all(0));
		}

		virtual void Create(NEAT::Genome& individual) {
			// create network
			individual.BuildPhenotype(NeuralNetwork);

			// calculate network depth
			individual.CalculateDepth();
			NetworkDepth = individual.GetDepth();

			// number of activations for this network 
			const int numActivations = (NetworkActivations < 1) ? NetworkDepth : NetworkActivations;

			// bias
			input[3] = 1.0;

			// create image
			for (int i = 0; i <= maxValue; ++i) {
				uchar * const p = Image.ptr<uchar>(i); // pointer to the current image's row
				for (int j = 0; j <= maxValue; ++j) {
					// x coordinate
					input[0] = TransformInput(i, maxValue);
					// y coordinate
					input[1] = TransformInput(j, maxValue);
					// distance from the central of the image
					input[2] = TransformInput(std::sqrt((i - halfMaxValue) * (i - halfMaxValue) + (j - halfMaxValue) * (j - halfMaxValue)), maxDistanceFromCenter);

					// evaluate the neural network
					NeuralNetwork.Flush();
					NeuralNetwork.Input(input);
					for(int r = 0; r < numActivations; ++r) NeuralNetwork.Activate();

					// set the resulting value to the image's pixel
					p[j] = uchar(std::round(TransformOutput(NeuralNetwork.Output()[0])));
				}
			}
		}
	};



	/// AGGREGATION FUNCTIONS

	// Uses only one feature as the fitness value.
	struct SelectOne : public ComponentHelper<AggregationFunction> {
		// Index of the selected feature.
		int FeatureIndex;

		SelectOne(int featureIndex) : FeatureIndex(featureIndex) {}

		virtual void Aggregate(const std::vector<double>& values) {
			Fitness = ScaledFitness = values[FeatureIndex];
		}
	};

	// Sums values from all features.
	struct Sum : public ComponentHelper<AggregationFunction> {
		virtual void Aggregate(const std::vector<double>& values) {
			double sum = 0;
			for(int i = 0; i < values.size(); ++i) sum += values[i];
			Fitness = ScaledFitness = sum;
		}
	};

	// Sums values from all features and penalizes them by penalties.
	struct SumWithPenalties : public ComponentHelper<AggregationFunction> {
		// Number of Feature functions to penalize an individual (negative feature).
		// They must be the last features in algorithm's features.
		const int NumFeaturesPenalties;

		SumWithPenalties(int numFeaturesPenalties) : NumFeaturesPenalties(numFeaturesPenalties) {}

		virtual void Aggregate(const std::vector<double>& values) {
			double sum = 0;

			// features
			for(int i = 0; i < values.size() - NumFeaturesPenalties; ++i) sum += values[i];

			// penalties
			for(int i = values.size() - NumFeaturesPenalties; i < values.size(); ++i) sum *= values[i];
			
			Fitness = ScaledFitness = sum;
		}
	};

	// Sums adaptively scaled values from all features.
	// Each feature is scaled to domain [0,1], thus each feature has the same weight in the final fitness.
	// The feature's value is scaled according to the global min and max values from the last iteration of the algorithm.
	class AdaptiveScalingSum : public ComponentHelper<AggregationFunction, AfterPopulationEvaluationAction, InitAction> {
	protected:
		std::vector<double> min, max, newMin, newMax; // min and max values of all features
		bool afterFirstRun = false; // do not scale for the first time, only set min/max values

	public:
		virtual void Init(AlgorithmTemplate& algorithm) {
			const int numFeatures = algorithm.Features.size();

			min.resize(numFeatures, std::numeric_limits<double>::max());
			max.resize(numFeatures, std::numeric_limits<double>::min());
			newMin.resize(numFeatures, std::numeric_limits<double>::max());
			newMax.resize(numFeatures, std::numeric_limits<double>::min());
		}

		virtual void Aggregate(const std::vector<double>& values) {
			double sum = 0, scaledSum = 0;
			for(int i = 0; i < values.size(); ++i) {
				const double value = values[i];

				// update new min/max values
				if(value > newMax[i]) newMax[i] = value;
				if(value < newMin[i]) newMin[i] = value;

				sum += value;
				scaledSum += (value - min[i]) / (max[i] - min[i]);
			}
			
			Fitness = sum;
			ScaledFitness = afterFirstRun ? scaledSum : sum;

			afterFirstRun = true;
		}

		virtual void AfterPopulationEvaluation(AlgorithmTemplate& algorithm) {
			// update min/max values of features
			for(int i = 0; i < min.size(); ++i) {
				min[i] = newMin[i];
				max[i] = newMax[i];
			}
		}
	};

	// Fitness value is a scaled distance to the given image in the feature space.
	// Each feature is adaptively scaled to domain [0,1]. Scaling works the same way as in AdaptiveScalingSum class.
	struct AdaptiveScalingDistanceInFeatureSpace : public AdaptiveScalingSum {
		cv::Mat OriginalImage; // original image
		std::vector<double> OriginalImageValues; // feature values of the original image

		AdaptiveScalingDistanceInFeatureSpace(const cv::Mat& image) : OriginalImage(image.clone()) {}

		virtual void Init(AlgorithmTemplate& algorithm) {
			AdaptiveScalingSum::Init(algorithm);

			// set feature values of the original image
			for(auto& feature : algorithm.Features) {
				double value = feature->Evaluate(OriginalImage);
				if(!std::isfinite(value)) value = 0.0;
				OriginalImageValues.push_back(value);
			}
		}

		virtual void Aggregate(const std::vector<double>& values) {
			double sum = 0, scaledSum = 0;
			for(int i = 0; i < values.size(); ++i) {
				const double value = abs(values[i] - OriginalImageValues[i]);
				const double scaledValue = value / max[i];

				// update new max value
				if(value > newMax[i]) newMax[i] = value;

				sum += value * value;
				scaledSum += scaledValue * scaledValue;
			}

			Fitness = 1.0 / sqrt(sum);
			ScaledFitness = 1.0 / sqrt(afterFirstRun ? scaledSum : sum);

			afterFirstRun = true;
		}
	};

	// Sums adaptively scaled values from all features and penalizes them by penalties.
	// Scaling works the same way as in AdaptiveScalingSum class.
	struct AdaptiveScalingSumWithPenalties : public AdaptiveScalingSum {
		// Number of Feature functions to penalize an individual (negative feature).
		// They must be the last features in algorithm's features.
		const int NumFeaturesPenalties;

		AdaptiveScalingSumWithPenalties(int numFeaturesPenalties) : NumFeaturesPenalties(numFeaturesPenalties) {}

		virtual void Aggregate(const std::vector<double>& values) {
			double sum = 0, scaledSum = 0;
			for(int i = 0; i < values.size() - NumFeaturesPenalties; ++i) {
				const double value = values[i];

				// update new min/max values
				if(value > newMax[i]) newMax[i] = value;
				if(value < newMin[i]) newMin[i] = value;

				sum += value;
				scaledSum += (value - min[i]) / (max[i] - min[i]);
			}

			// penalties
			for(int i = values.size() - NumFeaturesPenalties; i < values.size(); ++i) {
				const double value = values[i];

				// update new min/max values
				if(value > newMax[i]) newMax[i] = value;
				if(value < newMin[i]) newMin[i] = value;

				sum *= value;
				scaledSum *= value;
			}
			
			Fitness = sum;
			ScaledFitness = afterFirstRun ? scaledSum : sum;

			afterFirstRun = true;
		}
	};



	/// STATISTICS AND INFORMATION ABOUT THE ALGORITHM'S RUN

	// Stores fitness and all features' values of the best individual of every iteration.
	class BestIndividualStatistics : public ComponentHelper<AfterPopulationEvaluationAction> {

		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive& ar, const unsigned int version) {
			ar & BestIndividuals;
			ar & BestFeatures;
			ar & BestFitness;
			ar & BestScaledFitness;
		}

	public:
		// Best individuals
		std::vector<NEAT::Genome> BestIndividuals;
		// Feature values of the best individuals
		std::vector<std::vector<double>> BestFeatures;
		// Fitness values of the best individuals
		std::vector<double> BestFitness;
		// Scaled-fitness values of the best individuals
		std::vector<double> BestScaledFitness;

		virtual void AfterPopulationEvaluation(AlgorithmTemplate& algorithm) {
			const NEAT::Genome bestIndividual = algorithm.Population->GetBestGenome();

			BestIndividuals.push_back(bestIndividual);
			BestFeatures.push_back(bestIndividual.fitness);
			BestFitness.push_back(bestIndividual.GetFitness());
			BestScaledFitness.push_back(bestIndividual.GetScaledFitness());
		}

		// Saves all best individuals into separate files.
		void SaveBestIndividuals(const std::string& filename) {
			for(int i = 0; i < BestIndividuals.size(); ++i) {
				BestIndividuals[i].Save(Format("%1%%2%", filename, i).c_str());
			}
		}

		// Saves information about features' values to the file.
		void SaveBestFeatures(const std::string& filename) const {
			SaveAsText(BestFeatures, filename);
		}

		// Saves information about fitness value to the file.
		void SaveBestFitness(const std::string& filename) const {
			SaveAsText(BestFitness, filename);
		}

		void Save(const std::string& filename) const {
			std::ofstream file(filename);
			boost::archive::binary_oarchive oa(file);

			oa << *this;
		}

		void Load(const std::string& filename) {
			std::ifstream file(filename);
			boost::archive::binary_iarchive ia(file);

			ia >> *this;
		}
	};

	// Stores all information needed for aterwards statistics
	class InformationForStatistics : public ComponentHelper<AfterPopulationEvaluationAction> {

		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive& ar, const unsigned int version) {
			ar & FitnessValue;
			ar & ScaledFitnessValue;
			ar & FeaturesValues;
			ar & Stagnations;
			ar & BestIndividualIndex;
			ar & StoreIndividuals;
			ar & Individuals;
		}

	public:
		std::vector<std::vector<double>> FitnessValue;
		std::vector<std::vector<double>> ScaledFitnessValue;
		std::vector<std::vector<std::vector<double>>> FeaturesValues;
		std::vector<int> Stagnations;
		std::vector<int> BestIndividualIndex;

		bool StoreIndividuals = true;
		std::vector<std::vector<NEAT::Genome>> Individuals;

		virtual void AfterPopulationEvaluation(AlgorithmTemplate& algorithm) {
			std::vector<double> fitness;
			std::vector<double> scaledFitness;
			std::vector<std::vector<double>> features;
			std::vector<NEAT::Genome> individuals;

			double maxValue = std::numeric_limits<double>::min();
			int maxValueIndex = -1;
			int index = 0;

			for(int speciesIndex = 0; speciesIndex < algorithm.Population->m_Species.size(); ++speciesIndex) {
				for(int individualIndex = 0; individualIndex < algorithm.Population->m_Species[speciesIndex].m_Individuals.size(); ++individualIndex) {
					const NEAT::Genome individual = algorithm.Population->m_Species[speciesIndex].m_Individuals[individualIndex];
					
					fitness.push_back(individual.GetFitness());
					scaledFitness.push_back(individual.GetScaledFitness());
					features.push_back(individual.fitness);

					if(individual.GetScaledFitness() > maxValue) {
						maxValue = individual.GetScaledFitness();
						maxValueIndex = index;
					}

					if(StoreIndividuals) {
						individuals.push_back(individual);
					}

					++index;
				}
			}

			FitnessValue.push_back(fitness);
			ScaledFitnessValue.push_back(scaledFitness);
			FeaturesValues.push_back(features);
			
			Stagnations.push_back(algorithm.Population->GetStagnation());
			BestIndividualIndex.push_back(maxValueIndex);

			if(StoreIndividuals) Individuals.push_back(individuals);
		}

		void Save(const std::string& filename) const {
			std::ofstream file(filename);
			boost::archive::binary_oarchive oa(file);

			oa << *this;
		}

		void Load(const std::string& filename) {
			std::ifstream file(filename);
			boost::archive::binary_iarchive ia(file);

			ia >> *this;
		}
	};

	// Stores additional features values for the whole population.
	class FeaturesStatistics : public ComponentHelper<AfterIndividualEvaluationAction, AfterPopulationEvaluationAction> {
		std::vector<std::vector<double>> currentPopulation;

		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive& ar, const unsigned int version) {
			ar & FeaturesValues;
		}

	public:
		// Feature functions to evaluate an individual.
		std::vector<FeatureFunctionPtr> Features;
		std::vector<std::vector<std::vector<double>>> FeaturesValues;

		virtual void AfterIndividualEvaluation(AlgorithmTemplate& algorithm) {
			std::vector<double> features;

			for(int featureIndex = 0; featureIndex < Features.size(); ++featureIndex) {
				const double featureValue = Features[featureIndex]->Evaluate(algorithm.PhenotypeFunction->Image);
				features.push_back(featureValue);
			}

			currentPopulation.push_back(features);
		}

		virtual void AfterPopulationEvaluation(AlgorithmTemplate& algorithm) {
			FeaturesValues.push_back(currentPopulation);
			currentPopulation.clear();
		}

		void Save(const std::string& filename) const {
			std::ofstream file(filename);
			boost::archive::binary_oarchive oa(file);

			oa << *this;
		}

		void Load(const std::string& filename) {
			std::ifstream file(filename);
			boost::archive::binary_iarchive ia(file);

			ia >> *this;
		}
	};



	/// DEFINITION OF ATTACHING THE COMPONENT TO THE ALGORIHTM 

	template<>
	struct AutomaticAttach<PhenotypeFunction> {
		static void Attach(AlgorithmTemplate& a, PhenotypeFunctionPtr c) {
			a.PhenotypeFunction = c;
		}
	};

	template<>
	struct AutomaticAttach<FeatureFunction> {
		static void Attach(AlgorithmTemplate& a, FeatureFunctionPtr c) {
			a.Features.push_back(c);
		}
	};

	template<>
	struct AutomaticAttach<AggregationFunction> {
		static void Attach(AlgorithmTemplate& a, AggregationFunctionPtr c) {
			a.AggregationFunction = c;
		}
	};

	template<>
	struct AutomaticAttach<BeforePopulationEvaluationAction> {
		static void Attach(AlgorithmTemplate& a, BeforePopulationEvaluationActionPtr c) {
			a.BeforePopulationEvaluationActions.push_back(c);
		}
	};

	template<>
	struct AutomaticAttach<AfterPopulationEvaluationAction> {
		static void Attach(AlgorithmTemplate& a, AfterPopulationEvaluationActionPtr c) {
			a.AfterPopulationEvaluationActions.push_back(c);
		}
	};

	template<>
	struct AutomaticAttach<BeforeIndividualEvaluationAction> {
		static void Attach(AlgorithmTemplate& a, BeforeIndividualEvaluationActionPtr c) {
			a.BeforeIndividualEvaluationActions.push_back(c);
		}
	};

	template<>
	struct AutomaticAttach<AfterIndividualEvaluationAction> {
		static void Attach(AlgorithmTemplate& a, AfterIndividualEvaluationActionPtr c) {
			a.AfterIndividualEvaluationActions.push_back(c);
		}
	};

	template<>
	struct AutomaticAttach<InitAction> {
		static void Attach(AlgorithmTemplate& a, InitActionPtr c) {
			a.InitActions.push_back(c);
		}
	};

}

#endif	/* CORE_HPP */
