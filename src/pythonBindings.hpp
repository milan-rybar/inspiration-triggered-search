#ifndef PYTHONBINDINGS_HPP
#define	PYTHONBINDINGS_HPP

#include <vector>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "pyboostcvconverter/CVBoostConverter.hpp"

#include "core.hpp"
#include "features.hpp"
#include "its.hpp"

static void init_ar()
{
	Py_Initialize();
	import_array();
}

BOOST_PYTHON_MODULE(its) {
	using namespace boost::python;
	using namespace std;
	using namespace its;

	// automatic usage of cv::Image from Python (define automatic conversion for C++ <=> Python)

	init_ar();

	// initialize converters
	to_python_converter<cv::Mat, bcvt::matToNDArrayBoostConverter>();
	bcvt::matFromNDArrayBoostConverter();


	// CORE

	class_<Component, ComponentPtr, boost::noncopyable>("Component", no_init);

	class_<PhenotypeFunction, PhenotypeFunctionPtr, bases<Component>, boost::noncopyable>("PhenotypeFunction", no_init)
		.def_readonly("NeuralNetwork", &PhenotypeFunction::NeuralNetwork)
		.add_property("Image", &PhenotypeFunction::GetImage)

		.def("Create", &PhenotypeFunction::Create)
	;

	class_<FeatureFunction, FeatureFunctionPtr, bases<Component>, boost::noncopyable>("FeatureFunction", no_init)
		.def("Evaluate", &FeatureFunction::Evaluate)
	;
	class_<vector<FeatureFunctionPtr>>("FeatureFunctionVector").def(vector_indexing_suite<vector<FeatureFunctionPtr>, true>());

	class_<AggregationFunction, AggregationFunctionPtr, bases<Component>, boost::noncopyable>("AggregationFunction", no_init)
		.def_readonly("Fitness", &AggregationFunction::Fitness)
		.def_readonly("ScaledFitness", &AggregationFunction::ScaledFitness)

		.def("Aggregate", &AggregationFunction::Aggregate)
	;

	class_<BeforePopulationEvaluationAction, BeforePopulationEvaluationActionPtr, bases<Component>, boost::noncopyable>("BeforePopulationEvaluationAction", no_init)
		.def("BeforePopulationEvaluation", &BeforePopulationEvaluationAction::BeforePopulationEvaluation)
	;
	class_<vector<BeforePopulationEvaluationActionPtr>>("BeforePopulationEvaluationActionVector").def(vector_indexing_suite<vector<BeforePopulationEvaluationActionPtr>, true>());

	class_<AfterPopulationEvaluationAction, AfterPopulationEvaluationActionPtr, bases<Component>, boost::noncopyable>("AfterPopulationEvaluationAction", no_init)
		.def("AfterPopulationEvaluation", &AfterPopulationEvaluationAction::AfterPopulationEvaluation)
	;
	class_<vector<AfterPopulationEvaluationActionPtr>>("AfterPopulationEvaluationActionVector").def(vector_indexing_suite<vector<AfterPopulationEvaluationActionPtr>, true>());

	class_<BeforeIndividualEvaluationAction, BeforeIndividualEvaluationActionPtr, bases<Component>, boost::noncopyable>("BeforeIndividualEvaluationAction", no_init)
		.def("BeforeIndividualEvaluation", &BeforeIndividualEvaluationAction::BeforeIndividualEvaluation)
	;
	class_<vector<BeforeIndividualEvaluationActionPtr>>("BeforeIndividualEvaluationActionVector").def(vector_indexing_suite<vector<BeforeIndividualEvaluationActionPtr>, true>());

	class_<AfterIndividualEvaluationAction, AfterIndividualEvaluationActionPtr, bases<Component>, boost::noncopyable>("AfterIndividualEvaluationAction", no_init)
		.def("AfterIndividualEvaluation", &AfterIndividualEvaluationAction::AfterIndividualEvaluation)
	;
	class_<vector<AfterIndividualEvaluationActionPtr>>("AfterIndividualEvaluationActionVector").def(vector_indexing_suite<vector<AfterIndividualEvaluationActionPtr>, true>());

	class_<InitAction, InitActionPtr, bases<Component>, boost::noncopyable>("InitAction", no_init)
		.def("Init", &InitAction::Init)
	;
	class_<vector<InitActionPtr>>("InitActionVector").def(vector_indexing_suite<vector<InitActionPtr>, true>());


	class_<AlgorithmTemplate>("AlgorithmTemplate")
		.def_readwrite("Population", &AlgorithmTemplate::Population)
		.def_readwrite("Parameters", &AlgorithmTemplate::Parameters)

		.def_readwrite("PhenotypeFunction", &AlgorithmTemplate::PhenotypeFunction)
		.def_readwrite("Features", &AlgorithmTemplate::Features)
		.def_readwrite("AggregationFunction", &AlgorithmTemplate::AggregationFunction)
		.def_readwrite("BeforePopulationEvaluationActions", &AlgorithmTemplate::BeforePopulationEvaluationActions)
		.def_readwrite("AfterPopulationEvaluationActions", &AlgorithmTemplate::AfterPopulationEvaluationActions)
		.def_readwrite("BeforeIndividualEvaluationActions", &AlgorithmTemplate::BeforeIndividualEvaluationActions)
		.def_readwrite("AfterIndividualEvaluationActions", &AlgorithmTemplate::AfterIndividualEvaluationActions)
		.def_readwrite("InitActions", &AlgorithmTemplate::InitActions)

		.def_readonly("Generation", &AlgorithmTemplate::Generation)

		.def("Attach", &AlgorithmTemplate::Attach)
		.def("Init", &AlgorithmTemplate::Init)
		.def("EvaluateIndividual", &AlgorithmTemplate::EvaluateIndividual)
		.def("EvaluatePopulation", &AlgorithmTemplate::EvaluatePopulation)
		.def("RunOneGeneration", &AlgorithmTemplate::RunOneGeneration)
		.def("RunUntilStagnation", &AlgorithmTemplate::RunUntilStagnation)
		.def("RunGenerations", &AlgorithmTemplate::RunGenerations)
		.def("ResetStagnation", &AlgorithmTemplate::ResetStagnation)
	;


	// phenotype functions

	class_<ImagePhenotype, bases<PhenotypeFunction>>("ImagePhenotype", init<int>())
		.def_readwrite("NetworkActivations", &ImagePhenotype::NetworkActivations)
		.def_readonly("NetworkDepth", &ImagePhenotype::NetworkDepth)
	;


	// aggregation functions

	class_<SelectOne, bases<AggregationFunction>>("SelectOne", init<int>());
	class_<Sum, bases<AggregationFunction>>("Sum");
	class_<AdaptiveScalingSum, bases<AggregationFunction, AfterPopulationEvaluationAction, InitAction>>("AdaptiveScalingSum");
	class_<AdaptiveScalingDistanceInFeatureSpace, bases<AdaptiveScalingSum>>("AdaptiveScalingDistanceInFeatureSpace", init<cv::Mat>());
	class_<SumWithPenalties, bases<AggregationFunction>>("SumWithPenalties", init<int>());
	class_<AdaptiveScalingSumWithPenalties, bases<AdaptiveScalingSum>>("AdaptiveScalingSumWithPenalties", init<int>());
	

	// feature functions

	class_<Mean, bases<FeatureFunction>>("Mean");
	class_<Std, bases<FeatureFunction>>("Std");
	class_<DFTMean, bases<FeatureFunction>>("DFTMean");
	class_<DFTStd, bases<FeatureFunction>>("DFTStd");
	class_<DCTMean, bases<FeatureFunction>>("DCTMean");
	class_<DCTStd, bases<FeatureFunction>>("DCTStd");
	class_<MaxAbsLaplacian, bases<FeatureFunction>>("MaxAbsLaplacian");
	class_<Tenengrad, bases<FeatureFunction>>("Tenengrad");
	class_<NormalizedVariance, bases<FeatureFunction>>("NormalizedVariance");
	class_<Choppiness, bases<FeatureFunction>>("Choppiness");
	class_<StrictSymmetry, bases<FeatureFunction>>("StrictSymmetry");
	class_<RelaxedSymmetry, bases<FeatureFunction>>("RelaxedSymmetry");
	class_<GlobalContrastFactor, bases<FeatureFunction>>("GlobalContrastFactor");
	class_<JpegImageComplexity, bases<FeatureFunction>>("JpegImageComplexity");
	class_<DistanceInPixelSpace, bases<FeatureFunction>>("DistanceInPixelSpace", init<cv::Mat>());
	class_<Constant, bases<FeatureFunction>>("Constant");


	// penalties (negative features)

	class_<JpegImageComplexityPenalty, bases<FeatureFunction>>("JpegImageComplexityPenalty");
	class_<ChoppinessPenalty, bases<FeatureFunction>>("ChoppinessPenalty");


	// statistics

	class_<BestIndividualStatistics, bases<AfterPopulationEvaluationAction>>("BestIndividualStatistics")
		.def_readonly("BestIndividuals", &BestIndividualStatistics::BestIndividuals)
		.def_readonly("BestFeatures", &BestIndividualStatistics::BestFeatures)
		.def_readonly("BestFitness", &BestIndividualStatistics::BestFitness)
		.def_readonly("BestScaledFitness", &BestIndividualStatistics::BestScaledFitness)

		.def("SaveBestIndividuals", &BestIndividualStatistics::SaveBestIndividuals)
		.def("SaveBestFeatures", &BestIndividualStatistics::SaveBestFeatures)
		.def("SaveBestFitness", &BestIndividualStatistics::SaveBestFitness)
		.def("Save", &BestIndividualStatistics::Save)
		.def("Load", &BestIndividualStatistics::Load)
	;

	class_<InformationForStatistics, bases<AfterPopulationEvaluationAction>>("InformationForStatistics")
		.def_readonly("FitnessValue", &InformationForStatistics::FitnessValue)
		.def_readonly("ScaledFitnessValue", &InformationForStatistics::ScaledFitnessValue)
		.def_readonly("FeaturesValues", &InformationForStatistics::FeaturesValues)
		.def_readonly("Stagnations", &InformationForStatistics::Stagnations)
		.def_readonly("BestIndividualIndex", &InformationForStatistics::BestIndividualIndex)

		.def_readwrite("StoreIndividuals", &InformationForStatistics::StoreIndividuals)
		.def_readonly("Individuals", &InformationForStatistics::Individuals)

		.def("Save", &InformationForStatistics::Save)
		.def("Load", &InformationForStatistics::Load)
	;

	class_<FeaturesStatistics, bases<AfterIndividualEvaluationAction, AfterPopulationEvaluationAction>>("FeaturesStatistics")
		.def_readwrite("Features", &FeaturesStatistics::Features)
		.def_readonly("FeaturesValues", &FeaturesStatistics::FeaturesValues)

		.def("Save", &FeaturesStatistics::Save)
		.def("Load", &FeaturesStatistics::Load)
	;


	// ITS

	class_<Window>("Window")
		.def_readonly("Id", &Window::Id)
		.def_readonly("Rect", &Window::Rect)
		.def_readonly("Area", &Window::Area)
		.def_readonly("Value", &Window::Value)
		.def_readonly("C", &Window::C)
		.def_readonly("E", &Window::E)
		.def_readonly("EValues", &Window::EValues)
	;
	class_<std::vector<Window>>("WindowVector").def(vector_indexing_suite<std::vector<Window>>());

	class_<View>("View")
		.def_readonly("Area", &View::Area)
		.def_readonly("Windows", &View::Windows)
	;
	class_<std::vector<View>>("ViewVector").def(vector_indexing_suite<std::vector<View>>());

	class_<ObjectiveFunction>("ObjectiveFunction")
		.def_readonly("C", &ObjectiveFunction::C)
		.def_readonly("E", &ObjectiveFunction::E)
		.def_readonly("Views", &ObjectiveFunction::Views)
	;
	class_<std::vector<ObjectiveFunction>>("ObjectiveFunctionVector").def(vector_indexing_suite<std::vector<ObjectiveFunction>>());

	class_<Description, bases<ObjectiveFunction>>("Description")
		.def_readonly("MetricValue", &Description::MetricValue)
		.def_readonly("ObjectiveCValues", &Description::ObjectiveCValues)
	;
	class_<std::vector<Description>>("DescriptionVector").def(vector_indexing_suite<std::vector<Description>>());
	class_<std::vector<std::vector<Description>>>("DescriptionVector2D").def(vector_indexing_suite<std::vector<std::vector<Description>>>());


	class_<ViewFeatureFunction, ViewFeatureFunctionPtr, bases<Component>, boost::noncopyable>("ViewFeatureFunction", no_init)
		.def("Evaluate", &ViewFeatureFunction::Evaluate)
	;
	class_<vector<ViewFeatureFunctionPtr>>("ViewFeatureFunctionVector").def(vector_indexing_suite<vector<ViewFeatureFunctionPtr>, true>());

	class_<ModifyObjectiveOperator, ModifyObjectiveOperatorPtr, bases<Component>, boost::noncopyable>("ModifyObjectiveOperator", no_init)
		.def("ModifyObjective", &ModifyObjectiveOperator::ModifyObjective)
	;
	class_<vector<ModifyObjectiveOperatorPtr>>("ViewFeatureFunctionVector").def(vector_indexing_suite<vector<ModifyObjectiveOperatorPtr>, true>());

	class_<DescriptionToModifyObjectiveSelector, DescriptionToModifyObjectiveSelectorPtr, bases<Component>, boost::noncopyable>("DescriptionToModifyObjectiveSelector", no_init)
		.def("Select", &DescriptionToModifyObjectiveSelector::Select)
	;

	class_<DiversificationOperator, DiversificationOperatorPtr, bases<Component>, boost::noncopyable>("DiversificationOperator", no_init)
		.def("Diversify", &DiversificationOperator::Diversify)
		.def("UpdateAfterLastDiversification", &DiversificationOperator::UpdateAfterLastDiversification)
	;


	class_<ITS, bases<AlgorithmTemplate>>("ITS")
		.def_readwrite("StagnationLimit", &ITS::StagnationLimit)
		.def_readwrite("OptimizationIterations", &ITS::OptimizationIterations)
		.def_readwrite("InspirationCriterionThreshold", &ITS::InspirationCriterionThreshold)
		.def_readonly("Objective", &ITS::Objective)
		.def_readonly("Descriptions", &ITS::Descriptions)

		.def_readwrite("ViewFeatures", &ITS::ViewFeatures)
		.def_readwrite("ModifyObjectiveOperators", &ITS::ModifyObjectiveOperators)
		.def_readwrite("DescriptionToModifyObjective", &ITS::DescriptionToModifyObjective)
		.def_readwrite("Diversification", &ITS::Diversification)
		.def_readwrite("FeaturesPenalties", &ITS::FeaturesPenalties)

		.def_readonly("Evaluation", &ITS::Evaluation)
		.def_readonly("CreativeCycle", &ITS::CreativeCycle)

		.def("OneCreativeCycle", &ITS::OneCreativeCycle)
	;


	// view features

	class_<ContoursView, bases<ViewFeatureFunction>>("ContoursView");
	class_<ObjectDetectionView, bases<ViewFeatureFunction>>("ObjectDetectionView", init<std::string>());


	// modify objective operators

	class_<AddBestWindowByE, bases<ModifyObjectiveOperator>>("AddBestWindowByE");
	class_<AddRandomlyWindowByE, bases<ModifyObjectiveOperator>>("AddRandomlyWindowByE");
	class_<RemoveRandomlyWindowByC, bases<ModifyObjectiveOperator>>("RemoveRandomlyWindowByC")
		.def_readwrite("ThresholdToRemove", &RemoveRandomlyWindowByC::ThresholdToRemove)
	;


	// selector of description to modify objective

	class_<BestByMetric, bases<DescriptionToModifyObjectiveSelector>>("BestByMetric");
	class_<RandomlyByMetric, bases<DescriptionToModifyObjectiveSelector>>("RandomlyByMetric");


	// diversification operators 

	class_<MutatePopulation, bases<DiversificationOperator>>("MutatePopulation");
	class_<ConstantFitnessValue, bases<DiversificationOperator>>("ConstantFitnessValue");
	

	// statistics

	class_<ITSStatistics, bases<AfterPopulationEvaluationAction>>("ITSStatistics")
		.def_readonly("FitnessValue", &ITSStatistics::FitnessValue)
		.def_readonly("ScaledFitnessValue", &ITSStatistics::ScaledFitnessValue)
		.def_readonly("Stagnations", &ITSStatistics::Stagnations)
		.def_readonly("BestIndividualIndex", &ITSStatistics::BestIndividualIndex)

		.def_readonly("Phases", &ITSStatistics::Phases)
		.def_readonly("Objectives", &ITSStatistics::Objectives)
		.def_readonly("WindowAdded", &ITSStatistics::WindowAdded)
		.def_readonly("WindowFeatures", &ITSStatistics::WindowFeatures)

		.def_readwrite("StoreIndividuals", &ITSStatistics::StoreIndividuals)
		.def_readonly("Individuals", &ITSStatistics::Individuals)

		.def_readonly("Descriptions", &ITSStatistics::Descriptions)
		.def_readonly("BestDescriptionByMetricIndex", &ITSStatistics::BestDescriptionByMetricIndex)
		.def_readonly("DescriptionToModifyObjectiveIndex", &ITSStatistics::DescriptionToModifyObjectiveIndex)

		.def("Save", &ITSStatistics::Save)
		.def("Load", &ITSStatistics::Load)
	;

	enum_<ITSPhase>("ITSPhase")
		.value("OPTIMIZATION", ITSPhase::OPTIMIZATION)
		.value("DIVERSIFICATION", ITSPhase::DIVERSIFICATION)
		.value("INSPIRATION_CRITERION", ITSPhase::INSPIRATION_CRITERION)
		.value("MODIFY_OBJECTIVE", ITSPhase::MODIFY_OBJECTIVE)
		.value("SIMPLIFY_OBJECTIVE", ITSPhase::SIMPLIFY_OBJECTIVE)
	;
	class_<std::vector<ITSPhase>>("ITSPhaseVector").def(vector_indexing_suite<std::vector<ITSPhase>>());
	class_<std::vector<std::vector<ITSPhase>>>("ITSPhaseVector2D").def(vector_indexing_suite<std::vector<std::vector<ITSPhase>>>());


	// other

	class_<MutatePopulationEachGeneration, bases<BeforePopulationEvaluationAction>>("MutatePopulationEachGeneration");

	class_<std::vector<std::vector<std::vector<double>>>>("DoublesDoublesList2D")
		.def(vector_indexing_suite<std::vector<std::vector<std::vector<double>>>>())
	;

	class_<cv::Rect_<int>>("Rect")
		.def_readonly("x", &cv::Rect_<int>::x)
		.def_readonly("y", &cv::Rect_<int>::y)
		.def_readonly("width", &cv::Rect_<int>::width)
		.def_readonly("height", &cv::Rect_<int>::height)
	;
}

#endif	/* PYTHONBINDINGS_HPP */
