#ifndef ITS_HPP
#define	ITS_HPP

#include "core.hpp"

namespace its {

	/// CORE STRUCTURES FOR ITS ALGORITHM

	struct Window {
		int Id; // unique id
		cv::Rect Rect; // area
		double Area; // size of the area
		double Value; // fitness (feature) value in the particular space (view)
		double C, E; // the coverage and the extension of this window, when it makes sense
		std::vector<double> EValues; // parts of the E when computing against the objective function

		bool operator==(const Window& x) const { return Rect == x.Rect; } // needed for boost python
	};

	struct View {
		const static int size = 256;
		const static int fullSize = size * size;

		double Area; // size of the area = union of all windows
		std::vector<Window> Windows; // windows
		bool* ImageArea = nullptr; // for simple computing the size of the area by marking pixels in the image

		View() {}
		View(const View& other) : Area(other.Area), Windows(other.Windows), ImageArea(nullptr) {}
		View& operator=(const View& other) {
			Area = other.Area;
			Windows = other.Windows;
			ResetArea();
			std::cout<<"operator="<<std::endl;
			return *this;
		}

		~View() {
			if(ImageArea != nullptr) {
				delete[] ImageArea;
				ImageArea = nullptr;
			}
		}

		void ResetArea() {
			if(ImageArea == nullptr) ImageArea = new bool[fullSize];

			for(int i = 0; i < fullSize; ++i) {
				ImageArea[i] = false;
			}

			Area = 0;

			for(int i = 0; i < Windows.size(); ++i) {
				for(int y = Windows[i].Rect.y; y < Windows[i].Rect.y + Windows[i].Rect.height; ++y) {
					const int offset = size * y;
					for(int x = Windows[i].Rect.x; x < Windows[i].Rect.x + Windows[i].Rect.width; ++x) {
						const int index = offset + x;

						if(!ImageArea[index]) {
							ImageArea[index] = true;
							++Area;
						}
					}
				}
			}
		}

		int AreaOfIntersection(const Window& window) const {
			if(ImageArea == nullptr) return 0;

			int count = 0;

			for(int y = window.Rect.y; y < window.Rect.y + window.Rect.height; ++y) {
				const int offset = size * y;
				for(int x = window.Rect.x; x < window.Rect.x + window.Rect.width; ++x) {
					const int index = offset + x;
					if(ImageArea[index]) ++count;
				}
			}

			return count;
		}

		bool operator==(const View& x) const { return Windows == x.Windows; } // needed for boost python
	};

	struct ObjectiveFunction {
		double C, E; // the coverage and the extension of the description
		std::vector<View> Views; // views

		bool operator==(const ObjectiveFunction& x) const { return Views == x.Views; } // needed for boost python
	};

	struct Description : public ObjectiveFunction {
		std::vector<double> ObjectiveCValues; // parts of the C when computing against the objective function
		double MetricValue; // descriptin metric value

		bool operator==(const Description& x) const { return Views == x.Views; } // needed for boost python
	};


	// forward declaration of ITS 
	struct ITS;

	// View feature function to evaluate an image.
	struct ViewFeatureFunction : virtual public Component {
		virtual View Evaluate(const cv::Mat& image) = 0;
	};
	typedef boost::shared_ptr<ViewFeatureFunction> ViewFeatureFunctionPtr;


	// Operator to modify an objective function with the given description.  
	struct ModifyObjectiveOperator : virtual public Component {
		virtual void ModifyObjective(ITS& its, const Description& desc) = 0;
	};
	typedef boost::shared_ptr<ModifyObjectiveOperator> ModifyObjectiveOperatorPtr;


	// Selector of a description to modify an objective function. 
	struct DescriptionToModifyObjectiveSelector : virtual public Component {
		virtual int Select(ITS& its) = 0;
	};
	typedef boost::shared_ptr<DescriptionToModifyObjectiveSelector> DescriptionToModifyObjectiveSelectorPtr;


	// Operator to diversify the population.
	struct DiversificationOperator : virtual public Component {
		virtual int Diversify(ITS& its) = 0;
		virtual int UpdateAfterLastDiversification(ITS& its) = 0;
	};
	typedef boost::shared_ptr<DiversificationOperator> DiversificationOperatorPtr;


	/// ITS ALGORITHM

	// Phase of ITS (used only for statistics and visual purposes)
	enum class ITSPhase {
		OPTIMIZATION = 'o',
		DIVERSIFICATION = 'd',
		INSPIRATION_CRITERION = 'c',
		MODIFY_OBJECTIVE = 'm',
		SIMPLIFY_OBJECTIVE = 's'
	};

	// ITS algorithm
	struct ITS : public AlgorithmTemplate {
		const int size = 256; // image size = 256x256px
		const double fullSize = 256.0 * 256.0;

		// stagnation limit for NEAT
		int StagnationLimit = 10; 
		// number of iterations for the optimization (unless stagnation criterion is met)
		int OptimizationIterations = 10;
		// threshold for inspiration criterion
		double InspirationCriterionThreshold = 0.1;

		// View feature functions to evaluate an image.
		std::vector<ViewFeatureFunctionPtr> ViewFeatures;
		// Operators to modify the objective function.  
		std::vector<ModifyObjectiveOperatorPtr> ModifyObjectiveOperators;
		// Operator to select a description to modify an objective function.  
		DescriptionToModifyObjectiveSelectorPtr DescriptionToModifyObjective;
		// Operator to diversify the population.
		DiversificationOperatorPtr Diversification;

		// Feature functions to penalize an individual (negative feature).
		std::vector<FeatureFunctionPtr> FeaturesPenalties;

		std::vector<double> min, max, newMin, newMax; // min and max values of all features

		ObjectiveFunction Objective; // objective function
		std::vector<Description> Descriptions; // descriptions for the population

		double metricValueMax, metricValueSum; // metric information for the current descriptions
		int metricIndexMax;
		int descriptionIndexToModifyObjective;

		int windowCounter = 0; // for unique id of a window

		// Number of the current evaluation.
		int Evaluation = 0;
		// Number of the current "creative cycle".
		int CreativeCycle = 0;

		std::vector<ITSPhase> Phases;

		// Initializes the algorithm.
		virtual bool Init() {
			// checking of needed components
			if(ViewFeatures.size() == 0) {
				std::cerr << "No view features defined" << std::endl;
				return false;
			}
			if(ModifyObjectiveOperators.size() == 0) {
				std::cerr << "No operators to modify the objective function defined" << std::endl;
				return false;
			}
			if(DescriptionToModifyObjective == nullptr) {
				std::cerr << "No selector of description to modify objective defined" << std::endl;
				return false;
			}
			if(Diversification == nullptr) {
				std::cerr << "No operator to diversify the population" << std::endl;
				return false;
			}

			if(!initialized) {
				Attach(boost::make_shared<ImagePhenotype>(size));
				Attach(boost::make_shared<Sum>()); // only not to break AlgorithmTemplate, otherwise not needed
			}

			if(AlgorithmTemplate::Init()) {
				const int numFeatures = Features.size();

				// init values
				min.resize(numFeatures, std::numeric_limits<double>::max());
				max.resize(numFeatures, std::numeric_limits<double>::min());
				newMin.resize(numFeatures, std::numeric_limits<double>::max());
				newMax.resize(numFeatures, std::numeric_limits<double>::min());

				// create empty optimization technique
				for(int i = 0; i < numFeatures; ++i) {
					Objective.Views.push_back(View());
					Objective.Views[i].ResetArea();
				}

				BeforeFirstRun();

				return true;
			} else {
				return false;
			}
		}

		// Returns the area of the intersection between the given view and window.
		double AreaOfIntersection(const View& view, const Window& window) const {
			return view.AreaOfIntersection(window);
		}

		// Returns the area of the complement between the given view and window.
		double AreaOfComplement(const View& view, const Window& window) const {
			return window.Rect.area() - view.AreaOfIntersection(window);
		}

		// Computes the feature value in the specified area of the given image.
		double GetFeatureValue(const cv::Mat& image, const cv::Rect& rect, const int featureIndex) {
			const double value = Features[featureIndex]->Evaluate(cv::Mat(image, rect));

			// update new min/max values
			if(value > newMax[featureIndex]) newMax[featureIndex] = value;
			if(value < newMin[featureIndex]) newMin[featureIndex] = value;

			return value;
		}

		// Computes E for the current objective function and the given description.
		void ComputeE(Description& desc, const cv::Mat& image) {
			double descSumE = 0.0;
			int numWindows = 0;

			// go through all windows in the description
			for(int viewIndex = 0; viewIndex < desc.Views.size(); ++viewIndex) {
				for(int windowIndex = 0; windowIndex < desc.Views[viewIndex].Windows.size(); ++windowIndex) {

					double windowSumE = 0.0;

					// go through all views in the objective function
					for(int featureIndex = 0; featureIndex < Objective.Views.size(); ++featureIndex) {

						// area of complement between the window from the description and the current feature view from the objective function
						const double area = AreaOfComplement(Objective.Views[featureIndex], desc.Views[viewIndex].Windows[windowIndex]);

						// value of the window from the description in the current feature view
						double fvalue = GetFeatureValue(image, desc.Views[viewIndex].Windows[windowIndex].Rect, featureIndex);
						// scale this value to [0;1] by the min/max values for this feature from the last generation
						fvalue = ScaleFeatureValue(fvalue, featureIndex);

						// double wE = area * fvalue / (fullSize - Objective.Views[featureIndex].Area);
						// double wE = area * fvalue / desc.Views[viewIndex].Windows[windowIndex].Area;
						double wE = fvalue * 2.0 * area / (area + (fullSize - Objective.Views[featureIndex].Area));

						desc.Views[viewIndex].Windows[windowIndex].EValues.push_back(wE);

						windowSumE += wE;
					}

					// average extension in all features views/spaces 
					const double windowE = windowSumE / double(Objective.Views.size());

					desc.Views[viewIndex].Windows[windowIndex].E = windowE;

					descSumE += windowE;
					++numWindows;
				}
			}

			// average extension of all windows
			desc.E = (numWindows != 0) ? (descSumE / double(numWindows)) : 0.0;
		}

		// Computes C for the current objective function and the given description.
		void ComputeC(Description& desc) {
			double descSumC = 0.0;
			int numWindows = 0;

			desc.ObjectiveCValues.clear();

			// go through all windows in the objective function
			for(int featureIndex = 0; featureIndex < Objective.Views.size(); ++featureIndex) {
				for(int windowIndex = 0; windowIndex < Objective.Views[featureIndex].Windows.size(); ++windowIndex) {

					double windowSumC = 0.0;

					// go through all views in the description
					for(int viewIndex = 0; viewIndex < desc.Views.size(); ++viewIndex) {

						// area of intersection between the window from the objective function and the current view from the description
						const double area = AreaOfIntersection(desc.Views[viewIndex], Objective.Views[featureIndex].Windows[windowIndex]);

						// value of the window from the feature view the objective function in the current individual/description 
						double fvalue = Objective.Views[featureIndex].Windows[windowIndex].Value;
						// scale this value to [0;1] by the min/max values for this feature from the last generation
						fvalue = ScaleFeatureValue(fvalue, featureIndex);

						windowSumC += area * fvalue / Objective.Views[featureIndex].Windows[windowIndex].Area;
					}

					// average coverage in all views
					const double windowC = windowSumC / double(desc.Views.size());

					Objective.Views[featureIndex].Windows[windowIndex].C = windowC;
				
					descSumC += windowC;
					++numWindows;

					desc.ObjectiveCValues.push_back(windowC);
				}
			}

			// average coverage of all windows
			Objective.C = (numWindows != 0) ? (descSumC / double(numWindows)) : 1.0;
			desc.C = Objective.C;
		}

		// Computes metric value for the given description.
		inline double ComputeMetric(const Description& desc) const {
			return desc.C * desc.E;
		}

		// Creates a description for the given individual.
		Description CreateDescription(NEAT::Genome& individual) {
			Description desc;

			EvaluateIndividual(individual); // creates image and evaluates for the current objective function

			// create views of descriptions for the image
			for(int i = 0; i < ViewFeatures.size(); ++i) {
				desc.Views.push_back(ViewFeatures[i]->Evaluate(PhenotypeFunction->Image));
				desc.Views[i].ResetArea();
			}

			// compute E and C
			ComputeE(desc, PhenotypeFunction->Image);
			ComputeC(desc);

			// compute metric value for the description
			desc.MetricValue = ComputeMetric(desc);

			return desc;
		}

		// Returns whether the inspiration criterion is met or not.
		// Also computes the descriptions for the whole population.
		bool InspirationCriterion() {
			std::cout<<"Inspiration criterion"<<std::endl;
			Phases.push_back(ITSPhase::INSPIRATION_CRITERION);

			Descriptions.clear();

			metricValueMax = std::numeric_limits<double>::min();
			metricIndexMax = 0;
			metricValueSum = 0.0;

			for(int i = 0; i < BeforePopulationEvaluationActions.size(); ++i) BeforePopulationEvaluationActions[i]->BeforePopulationEvaluation(*this);

			// creates descriptions for all individuals in the population
			for(int speciesIndex = 0; speciesIndex < Population->m_Species.size(); ++speciesIndex) {
				for(int individualIndex = 0; individualIndex < Population->m_Species[speciesIndex].m_Individuals.size(); ++individualIndex) {
					Description desc = CreateDescription(Population->m_Species[speciesIndex].m_Individuals[individualIndex]);
					
					Population->m_Species[speciesIndex].m_Individuals[individualIndex].descriptionIndex = Descriptions.size();

					if(desc.MetricValue > metricValueMax) {
						metricValueMax = desc.MetricValue;
						metricIndexMax = Descriptions.size();
					}
					metricValueSum += desc.MetricValue;

					Descriptions.push_back(desc);
				}
			}

			// update min/max values of all features after the evaluation of the whole population
			UpdateMinMaxValues();

			for(int i = 0; i < AfterPopulationEvaluationActions.size(); ++i) AfterPopulationEvaluationActions[i]->AfterPopulationEvaluation(*this);

			// increase number of evaluation
			++Evaluation;
		
			// check inspiration criterion
			std::cout<<"Max metric value: "<<metricValueMax<<" of description index: "<<metricIndexMax<<std::endl;
			return metricValueMax >= InspirationCriterionThreshold;
		}

		// Inspiration phase
		// Modifies the current objective function.
		void InspirationPhase() {
			std::cout<<"Inspiration phase"<<std::endl;
			Phases.push_back(ITSPhase::MODIFY_OBJECTIVE);

			// choose description to define new objective function
			descriptionIndexToModifyObjective = DescriptionToModifyObjective->Select(*this);

			std::cout<<"Chosen description index: "<<descriptionIndexToModifyObjective<<std::endl;

			// define new objective function
			DefineNewObjectiveFunction(Descriptions[descriptionIndexToModifyObjective]);

			// update underlying optimization technique
			ResetStagnation();
		}

		// Modifies the current objective function with the given description.
		void DefineNewObjectiveFunction(const Description& desc) {
			for(int i = 0; i < ModifyObjectiveOperators.size(); ++i) {
				ModifyObjectiveOperators[i]->ModifyObjective(*this, desc);
			}
		}

		// Adds the given window to the current objective function (into the specified view).
		void AddWindowToObjectiveFunction(const Window& window, const int featureIndex) {
			Window w;
			w.Id = windowCounter++;
			w.Rect = window.Rect;
			w.Area = window.Area;

			// add window
			Objective.Views[featureIndex].Windows.push_back(w);
			// update area information
			Objective.Views[featureIndex].ResetArea();
		}

		// Diversification phase
		// Diversifies the population until the inspiration criterion is met 
		// and then modifies the current objective function.
		void DiversificationPhase() {
			std::cout<<"Diversification phase"<<std::endl;

			int tries = 0;

			while(true) {
				// diversify population
				std::cout<<"Diversify population"<<std::endl;
				Phases.push_back(ITSPhase::DIVERSIFICATION);

				// diversify the population
				Diversification->Diversify(*this);

				// inspiration criterion met
				if(InspirationCriterion()) {
					InspirationPhase();
					Diversification->UpdateAfterLastDiversification(*this);
					return;
				}
				// simplify objective function
				else if(Descriptions[0].ObjectiveCValues.size() > 0) {
					std::cout<<"Simplify objective"<<std::endl;
					Phases.push_back(ITSPhase::SIMPLIFY_OBJECTIVE);

					// remove a window with the lowest average C for the whole population

					// sum C values
					std::vector<double> C(Descriptions[0].ObjectiveCValues);

					for(int i = 1; i < Descriptions.size(); ++i) {
						for(int v = 1; v < C.size(); ++v) C[v] += Descriptions[i].ObjectiveCValues[v];
					}

					// find minimum C value
					double minValue = std::numeric_limits<double>::max();
					double minValueIndex = 0;

					for(int i = 1; i < C.size(); ++i) {
						if(C[i] < minValue || (C[i] == minValue && Population->m_RNG.RandFloat() > 0.5)) {
							minValue = C[i];
							minValueIndex = i;
						}
					}

					// remove this window
					RemoveWindowFromObjectiveFunction(minValueIndex);
				}
				// trying diversification for too long => start again from scratch (reset NEAT)
				else if(tries >= 10) {
					ResetNEAT();
					tries = 0;
				}
				else {
					++tries;
				}
				std::cout<<"Objective windows: "<<Descriptions[0].ObjectiveCValues.size()<<std::endl;
			}
		}

		// Removes the specified window from the current objective function.
		void RemoveWindowFromObjectiveFunction(const int windowIndex) {
			int count = windowIndex;
			for(int featureIndex = 0; featureIndex < Objective.Views.size(); ++featureIndex) {
				if(count < Objective.Views[featureIndex].Windows.size()) {
					// remove window
					Objective.Views[featureIndex].Windows.erase(Objective.Views[featureIndex].Windows.begin() + count);
				
					// update area information
					Objective.Views[featureIndex].ResetArea();

					return;
				}

				count -= Objective.Views[featureIndex].Windows.size();
			}
		}

		/// Evaluates the given individual (without calling the general actions).
		inline void EvaluateIndividualWiotActions(NEAT::Genome& individual) {
			// create phenotype
			PhenotypeFunction->Create(individual);

			// evaluate objective function
			individual.fitness.clear();

			double scaledFitness = 0.0;
			double unscaledFitness = 0.0;

			for(int featureIndex = 0; featureIndex < Objective.Views.size(); ++featureIndex) {
				for(int windowIndex = 0; windowIndex < Objective.Views[featureIndex].Windows.size(); ++windowIndex) {
					const double featureValue = GetFeatureValue(PhenotypeFunction->Image, Objective.Views[featureIndex].Windows[windowIndex].Rect, featureIndex);

					Objective.Views[featureIndex].Windows[windowIndex].Value = featureValue;
					individual.fitness.push_back(featureValue);

					unscaledFitness += featureValue;
					scaledFitness += ScaleFeatureValue(featureValue, featureIndex);
				}
			}

			// evaluate penalty
			double penalty = 1.0;
			for(int i = 0; i < FeaturesPenalties.size(); ++i) {
				penalty *= FeaturesPenalties[i]->Evaluate(PhenotypeFunction->Image);
			}

			// set individual's fitness value
			individual.SetFitness(unscaledFitness * penalty, scaledFitness * penalty);
			individual.SetEvaluated();
		}

		// Evaluates the given individual.
		virtual void EvaluateIndividual(NEAT::Genome& individual) {
			for(int i = 0; i < BeforeIndividualEvaluationActions.size(); ++i) BeforeIndividualEvaluationActions[i]->BeforeIndividualEvaluation(*this);

			EvaluateIndividualWiotActions(individual);

			for(int i = 0; i < AfterIndividualEvaluationActions.size(); ++i) AfterIndividualEvaluationActions[i]->AfterIndividualEvaluation(*this);
		}

		// Optimization phase
		// Optimizes the population by the current objective function.
		void OptimizationPhase() {
			int iteration = 0;

			do {
				// evaluate population
				EvaluatePopulation(*Population);

				std::cout<<"Optimization"<<std::endl;
				Phases.push_back(ITSPhase::OPTIMIZATION);

				// one iteration of NEAT
				Population->Epoch();

				// increase the number of generations
				++Generation;

				UpdateMinMaxValues();

				// increase number of evaluations
				++Evaluation;

				++iteration;
			} while(iteration < OptimizationIterations && Population->GetStagnation() <= StagnationLimit);
		}

		// Initialization phase before the first creative cycle. 
		void BeforeFirstRun() {
			// diversify population
			for(int i = 0; i < 3; ++i) Diversification->Diversify(*this);

			// evaluate all individuals in the population in order to get initial min/max values
			for(int speciesIndex = 0; speciesIndex < Population->m_Species.size(); ++speciesIndex) {
				for(int individualIndex = 0; individualIndex < Population->m_Species[speciesIndex].m_Individuals.size(); ++individualIndex) {
					// create phenotype
					PhenotypeFunction->Create(Population->m_Species[speciesIndex].m_Individuals[individualIndex]);

					// evaluate all features
					for(int featureIndex = 0; featureIndex < Features.size(); ++featureIndex) {
						GetFeatureValue(PhenotypeFunction->Image, cv::Rect(0, 0, size, size), featureIndex);
					}
				}
			}
			UpdateMinMaxValues();

			// set the first objective function
			DiversificationPhase();
		}

		// Does one creative cycle of ITS.
		void OneCreativeCycle() {
			// initialize the algorithm, if needed
			if(!initialized) {
				Init();
			}

			// k-iteration of optimization technique
			OptimizationPhase();

			// inspiration criterion => inspiration phase
			if(InspirationCriterion()) InspirationPhase();
			// stagnation criterion => diversification phase
			else if(Population->GetStagnation() > StagnationLimit) DiversificationPhase();

			// increase number of "creative cycle"
			++CreativeCycle;
		}

		void UpdateMinMaxValues() {
			// update min/max values of features
			for(int i = 0; i < min.size(); ++i) {
				min[i] = newMin[i];
				max[i] = newMax[i];
			}
		}

		// Returns the scaled value of the feature value.
		inline double ScaleFeatureValue(const double value, const int featureIndex) const {
			if(min[featureIndex] != std::numeric_limits<double>::max()) {
				return (value - min[featureIndex]) / (max[featureIndex] - min[featureIndex]);
			} else {
				return value;
			}
		}
	};



	/// VIEWS FEATURES

	// Detects contours in the image
	struct ContoursView : public ComponentHelper<ViewFeatureFunction> {
		virtual View Evaluate(const cv::Mat& image) {
			View view;

			const int threshold = 100;
			cv::Mat canny_output;
			std::vector<std::vector<cv::Point>> contours;
			std::vector<cv::Vec4i> hierarchy;

			/// Detect edges using canny
			cv::Canny(image, canny_output, threshold, threshold * 2, 3);
			/// Find contours
			cv::findContours(canny_output, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			for(int i = 0; i < contours.size(); ++i) {
				// ignore too small contours
				if(cv::contourArea(contours[i]) < 50) continue;

				// rectangle of the contours
				const cv::Rect rect = cv::boundingRect(contours[i]);
				// area of the rectangle
				const double area = rect.area();

				// ignore small or too big rectangles
				if(area > 2000.0 && area <= image.rows * image.cols / 4.0) {
					Window w;
					w.Rect = rect;
					w.Area = area;
					w.Value = 1.0;

					view.Windows.push_back(w);
				}
			}

			return view;
		}
	};

	// Detects an object by cascade classifier 
	class ObjectDetectionView : public ComponentHelper<ViewFeatureFunction> {
		cv::CascadeClassifier classifier;
	public:
		ObjectDetectionView(const std::string& filename) : classifier(filename) {
			if(classifier.empty()) {
				std::cerr<<"--(!)Error loading cascade"<<std::endl;
			}
		}

		virtual View Evaluate(const cv::Mat& image) {
			View view;

			if(classifier.empty()) {
				std::cerr<<"--(!)Error loading cascade"<<std::endl;
				return view;
			}

			std::vector<cv::Rect> objects;

			// detect objects
			classifier.detectMultiScale(image, objects, 1.1, 4);

			for (int i = 0; i < objects.size(); i++) {
				Window w;
				w.Rect = objects[i];
				w.Area = w.Rect.area();
				w.Value = 2.0;

				view.Windows.push_back(w);
			}

			return view;
		}
	};



	/// MODIFY OBJECTIVE OPERATORS

	// Adds the best window by E value.
	struct AddBestWindowByE : public ComponentHelper<ModifyObjectiveOperator> {
		virtual void ModifyObjective(ITS& its, const Description& desc) {
			double maxValue = std::numeric_limits<double>::min();
			int maxIndexView = -1, maxIndexWindow = -1, maxIndexFeatureView = -1;

			for(int i = 0; i < desc.Views.size(); ++i) {
				for(int j = 0; j < desc.Views[i].Windows.size(); ++j) {
					for(int viewIndex = 0; viewIndex < desc.Views[i].Windows[j].EValues.size(); ++viewIndex) {
						if(desc.Views[i].Windows[j].EValues[viewIndex] > maxValue || (desc.Views[i].Windows[j].EValues[viewIndex] == maxValue && its.Population->m_RNG.RandFloat() > 0.5)) {
							maxValue = desc.Views[i].Windows[j].EValues[viewIndex];
							maxIndexView = i;
							maxIndexWindow = j;
							maxIndexFeatureView = viewIndex;
						}
					}
				}
			}

			if(maxIndexView != -1) its.AddWindowToObjectiveFunction(desc.Views[maxIndexView].Windows[maxIndexWindow], maxIndexFeatureView);
		}
	};

	// Adds randomly a window by E value.
	struct AddRandomlyWindowByE : public ComponentHelper<ModifyObjectiveOperator> {
		virtual void ModifyObjective(ITS& its, const Description& desc) {
			// sum E values
			double ESum = 0.0;
			for(int i = 0; i < desc.Views.size(); ++i) {
				for(int j = 0; j < desc.Views[i].Windows.size(); ++j) {
					ESum += desc.Views[i].Windows[j].E * its.Objective.Views.size(); // change mean to sum of E in feature views
				}
			}

			// choose randomly a window by E value 
			double probability = its.Population->m_RNG.RandFloat() * ESum;

			for(int i = 0; i < desc.Views.size(); ++i) {
				for(int j = 0; j < desc.Views[i].Windows.size(); ++j) {
					const double wSumE = desc.Views[i].Windows[j].E * its.Objective.Views.size();

					// this window?
					if(probability <= wSumE) {
						for(int viewIndex = 0; viewIndex < desc.Views[i].Windows[j].EValues.size(); ++viewIndex) {
							probability -= desc.Views[i].Windows[j].EValues[viewIndex];

							// this feature view?
							if(probability <= 0.0) {
								its.AddWindowToObjectiveFunction(desc.Views[i].Windows[j], viewIndex);
								return;
							}
						}
					}

					probability -= wSumE;
				}
			}
		}
	};

	// Removes randomly a window by C value.
	struct RemoveRandomlyWindowByC : public ComponentHelper<ModifyObjectiveOperator> {
		// Window C value must be below this threshold to be considered for removal.
		double ThresholdToRemove = 0.5;

		virtual void ModifyObjective(ITS& its, const Description& desc) {
			for(int i = 0; i < desc.ObjectiveCValues.size(); ++i) {
				if(desc.ObjectiveCValues[i] < ThresholdToRemove && desc.ObjectiveCValues[i] < its.Population->m_RNG.RandFloat()) {
					std::cout<<"Remove window index: "<<i<<std::endl;

					// remove this window
					its.RemoveWindowFromObjectiveFunction(i);
					return;
				}
			}
		}
	};



	/// SELECTOR OF A DESCRIPTION TO MODIFY AN OBJECTIVE

	// Selects the best description by metric value.
	struct BestByMetric : public ComponentHelper<DescriptionToModifyObjectiveSelector> {
		virtual int Select(ITS& its) {
			return its.metricIndexMax;
		}
	};

	// Selects the best description by metric value.
	struct RandomlyByMetric : public ComponentHelper<DescriptionToModifyObjectiveSelector> {
		virtual int Select(ITS& its) {
			// assumption: descriptions for the population has been created, and metricValueSum is computed

			double probability = its.Population->m_RNG.RandFloat() * its.metricValueSum;

			for(int i = 0; i < its.Descriptions.size(); ++i) {
				probability -= its.Descriptions[i].MetricValue;	

				if(probability <= 0) {
					return i;
				}
			}

			return 0;
		}
	};



	/// DIVERSIFICATION OPERATORS

	// Mutates the population by NEAT mutations.
	struct MutatePopulation : public ComponentHelper<DiversificationOperator> {

		void MutateIndividuals(NEAT::Population& population) {
			for(int speciesIndex = 0; speciesIndex < population.m_Species.size(); ++speciesIndex) {
				for(int individualIndex = 0; individualIndex < population.m_Species[speciesIndex].m_Individuals.size(); ++individualIndex) {
					population.m_Species[speciesIndex].MutateGenome(false, population, population.m_Species[speciesIndex].m_Individuals[individualIndex], population.m_Parameters, population.m_RNG);
				}
			}
		}

		void UpdateNEATAfterExternalChanges(NEAT::Population& population) {
			population.m_Genomes.clear();

			for(int speciesIndex = 0; speciesIndex < population.m_Species.size(); ++speciesIndex) {
				for(int individualIndex = 0; individualIndex < population.m_Species[speciesIndex].m_Individuals.size(); ++individualIndex) {
					population.m_Genomes.push_back(population.m_Species[speciesIndex].m_Individuals[individualIndex]);
				}
			}

			population.m_Species.clear();

			// separates the population into species
			population.Speciate();
		}

		virtual int Diversify(ITS& its) {
			for(int i = 0; i < 2; ++i) MutateIndividuals(*its.Population);
		}

		virtual int UpdateAfterLastDiversification(ITS& its) {
			UpdateNEATAfterExternalChanges(*its.Population);
		}
	};

	// Runs NEAT with a constant fitness value = random walk.
	struct ConstantFitnessValue : public ComponentHelper<DiversificationOperator> {

		void EpochWithConstantFitness(NEAT::Population& population) {
			population.m_GensSinceBestFitnessLastChanged = 0;
			population.m_BestFitnessEver = 0;

			for(int speciesIndex = 0; speciesIndex < population.m_Species.size(); ++speciesIndex) {
				for(int individualIndex = 0; individualIndex < population.m_Species[speciesIndex].m_Individuals.size(); ++individualIndex) {
						population.m_Species[speciesIndex].m_Individuals[individualIndex].SetFitness(1.0, 1.0);
						population.m_Species[speciesIndex].m_Individuals[individualIndex].SetEvaluated();
				}
			}

			population.Epoch();
		}

		virtual int Diversify(ITS& its) {
			for(int i = 0; i < 2; ++i) EpochWithConstantFitness(*its.Population);
		}

		virtual int UpdateAfterLastDiversification(ITS& its) {
		}
	};



	/// STATISTICS

	struct ITSStatistics : public ComponentHelper<AfterPopulationEvaluationAction>  {

		// Fitness value for the whole population.
		std::vector<std::vector<double>> FitnessValue;
		// Scaled-fitness value for the whole population.
		std::vector<std::vector<double>> ScaledFitnessValue;
		// Number of stagnations.
		std::vector<int> Stagnations;
		// Index of the best individual (by scaled-fitness value).
		std::vector<int> BestIndividualIndex;

		// Phases in ITS.
		std::vector<std::vector<ITSPhase>> Phases;
		// Objective function.
		std::vector<ObjectiveFunction> Objectives;

		// Number of evaluation when windows were added to objective function.
		std::vector<int> WindowAdded;
		// Feature values for all windows.
		std::vector<std::vector<std::vector<double>>> WindowFeatures;

		// Store the whole population?
		bool StoreIndividuals = true;
		// The whole population.
		std::vector<std::vector<NEAT::Genome>> Individuals;

		// All descriptions.
		std::vector<std::vector<Description>> Descriptions;
		// Index of the best description by metric value.
		std::vector<int> BestDescriptionByMetricIndex;
		// Index of the description to modify objective function.
		std::vector<int> DescriptionToModifyObjectiveIndex;

		bool InspirationCriterion;
		bool ModifyObjective;

		virtual void AfterPopulationEvaluation(AlgorithmTemplate& algorithm) {
			ITS& its = static_cast<ITS&>(algorithm);

			// phases of ITS
			InspirationCriterion = false;
			ModifyObjective = false;
			for(int i = 0; i < its.Phases.size(); ++i) {
				if(its.Phases[i] == ITSPhase::INSPIRATION_CRITERION) InspirationCriterion = true;
				else if(its.Phases[i] == ITSPhase::MODIFY_OBJECTIVE) ModifyObjective = true;

				// std::cout<<(char)(its.Phases[i])<<' ';
			}
			// std::cout<<std::endl;
			// std::cout<<"Evaluation: "<<its.Evaluation<<std::endl;

			Phases.push_back(its.Phases);
			its.Phases.clear();

			// create indexes for windows
			std::vector<int> windowIndexes;

			for(int viewIndex = 0; viewIndex < its.Objective.Views.size(); ++viewIndex) {
				for(int windowIndex = 0; windowIndex < its.Objective.Views[viewIndex].Windows.size(); ++windowIndex) {
					const Window window = its.Objective.Views[viewIndex].Windows[windowIndex];

					if(window.Id >= WindowAdded.size()) {
						WindowAdded.resize(window.Id + 1, -1);
						WindowFeatures.resize(window.Id + 1);
					}
					if(WindowAdded[window.Id] == -1) {
						WindowAdded[window.Id] = its.Evaluation;
					}

					windowIndexes.push_back(window.Id);
				}
			}

			// get information from the population

			std::vector<double> fitness;
			std::vector<double> scaledFitness;
			std::vector<NEAT::Genome> individuals;
			std::vector<std::vector<double>> windowFeatures(windowIndexes.size());

			double maxValue = std::numeric_limits<double>::min();
			int maxValueIndex = -1;

			for(int speciesIndex = 0; speciesIndex < its.Population->m_Species.size(); ++speciesIndex) {
				for(int individualIndex = 0; individualIndex < its.Population->m_Species[speciesIndex].m_Individuals.size(); ++individualIndex) {
					const NEAT::Genome individual = its.Population->m_Species[speciesIndex].m_Individuals[individualIndex];
					const int index = fitness.size();

					fitness.push_back(individual.GetFitness());
					scaledFitness.push_back(individual.GetScaledFitness());

					if(individual.GetScaledFitness() > maxValue) {
						maxValue = individual.GetScaledFitness();
						maxValueIndex = index;
					}

					if(windowFeatures.size() > individual.fitness.size()) std::cerr << "Error: no information about all windows features" << std::endl;
					for(int i = 0; i < windowFeatures.size(); ++i) {
						windowFeatures[i].push_back(individual.fitness[i]);
					}

					if(StoreIndividuals) {
						individuals.push_back(individual);
					}
				}
			}

			// add features values to the correct windows
			for(int i = 0; i < windowIndexes.size(); ++i) {
				WindowFeatures[windowIndexes[i]].push_back(windowFeatures[i]);
			}

			FitnessValue.push_back(fitness);
			ScaledFitnessValue.push_back(scaledFitness);
			Stagnations.push_back(its.Population->GetStagnation());
			BestIndividualIndex.push_back(maxValueIndex);

			if(StoreIndividuals) Individuals.push_back(individuals);

			Objectives.push_back(its.Objective);

			Descriptions.push_back(InspirationCriterion || ModifyObjective ? its.Descriptions : std::vector<Description>());
			BestDescriptionByMetricIndex.push_back(InspirationCriterion || ModifyObjective ? its.metricIndexMax : -1);
			DescriptionToModifyObjectiveIndex.push_back(ModifyObjective ? its.descriptionIndexToModifyObjective : -1);
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

	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive& ar, const unsigned int version) {
			ar & FitnessValue;
			ar & ScaledFitnessValue;
			ar & Stagnations;
			ar & BestIndividualIndex;

			ar & Phases;
			ar & Objectives;

			ar & WindowAdded;
			ar & WindowFeatures;

			ar & StoreIndividuals;
			ar & Individuals;

			ar & Descriptions;
			ar & BestDescriptionByMetricIndex;
			ar & DescriptionToModifyObjectiveIndex;
		}
	};



	/// DEFINITION OF ATTACHING THE COMPONENT TO THE ALGORIHTM 

	template<>
	struct AutomaticAttach<ViewFeatureFunction> {
		static void Attach(AlgorithmTemplate& a, ViewFeatureFunctionPtr c) {
			static_cast<ITS&>(a).ViewFeatures.push_back(c);
		}
	};

	template<>
	struct AutomaticAttach<ModifyObjectiveOperator> {
		static void Attach(AlgorithmTemplate& a, ModifyObjectiveOperatorPtr c) {
			static_cast<ITS&>(a).ModifyObjectiveOperators.push_back(c);
		}
	};

	template<>
	struct AutomaticAttach<DescriptionToModifyObjectiveSelector> {
		static void Attach(AlgorithmTemplate& a, DescriptionToModifyObjectiveSelectorPtr c) {
			static_cast<ITS&>(a).DescriptionToModifyObjective = c;
		}
	};

	template<>
	struct AutomaticAttach<DiversificationOperator> {
		static void Attach(AlgorithmTemplate& a, DiversificationOperatorPtr c) {
			static_cast<ITS&>(a).Diversification = c;
		}
	};



	/// SPECIAL

	// Mutates NEAT population each generation.
	class MutatePopulationEachGeneration : public ComponentHelper<BeforePopulationEvaluationAction> {
		MutatePopulation mutator;

	public:
		virtual void BeforePopulationEvaluation(AlgorithmTemplate& algorithm) {
			for(int i = 0; i < 2; ++i) mutator.MutateIndividuals(*algorithm.Population);
			mutator.UpdateNEATAfterExternalChanges(*algorithm.Population);
		}
	};

}



/// SERIALIZATION OF ITS STRUCTURES

namespace boost {
	namespace serialization {

		template<class Archive>
		void serialize(Archive& ar, cv::Rect_<int>& rect, const unsigned int version) {
			ar & rect.x;
			ar & rect.y;
			ar & rect.width;
			ar & rect.height;
		}

		template<class Archive>
		void serialize(Archive& ar, its::Window& w, const unsigned int version) {
			ar & w.Id;
			ar & w.Rect;
			ar & w.Area;
			ar & w.Value;
			ar & w.C;
			ar & w.E;
		}
		
		template<class Archive>
		void serialize(Archive& ar, its::View& v, const unsigned int version) {
			ar & v.Area;
			ar & v.Windows;
		}

		template<class Archive>
		void serialize(Archive& ar, its::ObjectiveFunction& o, const unsigned int version) {
			ar & o.C;
			ar & o.E;
			ar & o.Views;
		}

		template<class Archive>
		void serialize(Archive& ar, its::Description& d, const unsigned int version) {
			ar & boost::serialization::base_object<its::ObjectiveFunction>(d);
			ar & d.MetricValue;
		}
	}
}

#endif	/* ITS_HPP */
