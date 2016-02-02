#ifndef FEATURES_HPP
#define	FEATURES_HPP

#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "core.hpp"

namespace its {
	
	// Mean
	struct Mean : public ComponentHelper<FeatureFunction> {
		virtual double Evaluate(const cv::Mat& image) {
			return cv::mean(image)[0];
		}
	};

	// Standard deviation
	class Std : public ComponentHelper<FeatureFunction> {
		cv::Scalar mean, std;
	public:
		virtual double Evaluate(const cv::Mat& image) {
			cv::meanStdDev(image, mean, std);
			return std[0];
		}
	};


	// Base class for DFT computation.
	class DFT {
	protected:
		cv::Mat complexI;
		cv::Mat magI;

		void ComputeDFT(const cv::Mat& image) {
			// expand image to the optimal size //with zero values on the border
			int m = cv::getOptimalDFTSize(image.rows);
			int n = cv::getOptimalDFTSize(image.cols);
			cv::Mat padded;
			cv::copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

			// convert to complex numbers
			cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
			cv::merge(planes, 2, complexI);

			// DFT
			cv::dft(complexI, complexI);

			// magnitude of the complex DFT
			cv::split(complexI, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
			cv::magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
			magI = planes[0];

			// crop to the original size
			magI = magI(cv::Rect(0, 0, image.rows, image.cols));
		}
	};

	// Mean of DFT
	class DFTMean : public DFT, public ComponentHelper<FeatureFunction> {
	protected:
		cv::Scalar dftMean;

	public:
		virtual double Evaluate(const cv::Mat& image) {
			ComputeDFT(image);
			dftMean = cv::mean(magI);
			return dftMean[0];
		}
	};

	// Std of DFT
	class DFTStd : public DFTMean {
		cv::Scalar dftStd;

	public:
		virtual double Evaluate(const cv::Mat& image) {
			ComputeDFT(image);
			cv::meanStdDev(magI, dftMean, dftStd);
			return dftStd[0];
		}
	};


	// Base class for DCT computation.
	class DCT {
	protected:
		cv::Mat freq;

		void ComputeDCT(const cv::Mat& image) {
			cv::Mat imageFloat = cv::Mat(image.rows, image.cols, CV_64F);
			image.convertTo(imageFloat, CV_64F);

			// DCT
			cv::dct(imageFloat, freq);
		}
	};

	// Mean of DCT
	class DCTMean : public DCT, public ComponentHelper<FeatureFunction> {
	protected:
		cv::Scalar dctMean;

	public:
		virtual double Evaluate(const cv::Mat& image) {
			ComputeDCT(image);
			dctMean = cv::mean(freq);
			return dctMean[0];
		}
	};

	// Std of DCT
	class DCTStd : public DCTMean {
		cv::Scalar dctStd;

	public:
		virtual double Evaluate(const cv::Mat& image) {
			ComputeDCT(image);
			cv::meanStdDev(freq, dctMean, dctStd);
			return dctStd[0];
		}
	};


	// Maximum of absolute Laplacian
	class MaxAbsLaplacian : public ComponentHelper<FeatureFunction> {
		cv::Mat laplacian;
		cv::Mat absLaplacian;

	public:
		virtual double Evaluate(const cv::Mat& image) {
			// Apply Laplace function
			cv::Laplacian(image, laplacian, CV_16S, 1);
			cv::convertScaleAbs(laplacian, absLaplacian);

			// find max value
			uchar maxValue = 0;
			for(int i = 0; i < image.rows * image.cols; ++i) {
				if(maxValue < absLaplacian.data[i]) maxValue = absLaplacian.data[i];
			}

			return maxValue;
		}
	};


	// Tenengrad algorithm
	class Tenengrad : public ComponentHelper<FeatureFunction> {
		cv::Mat Gx, Gy;

	public:
		virtual double Evaluate(const cv::Mat& image) {
			cv::Sobel(image, Gx, CV_64F, 1, 0, 3);
			cv::Sobel(image, Gy, CV_64F, 0, 1, 3);

			return cv::mean(Gx.mul(Gx) + Gy.mul(Gy)).val[0];
		}
	};


	// Normalized gray-level variance
	class NormalizedVariance : public ComponentHelper<FeatureFunction> {
		cv::Scalar mu, sigma;

	public:
		virtual double Evaluate(const cv::Mat& image) {
			cv::meanStdDev(image, mu, sigma);

			return (sigma.val[0] * sigma.val[0]) / mu.val[0];
		}
	};


	// Choppiness
	struct Choppiness : public ComponentHelper<FeatureFunction> {
		virtual double Evaluate(const cv::Mat& image) {
			const int neighbourhood = 5;

			double sumForChoppiness = 0.0;
			for(int rowIndex = 0; rowIndex < image.rows - neighbourhood; ++rowIndex) {
				for(int columnIndex = 0; columnIndex < image.cols - neighbourhood; ++columnIndex) {

					// mean of the window
					double sumForMean = 0.0;
					for(int i = 0; i < neighbourhood; ++i) {
						uchar const * p = image.ptr<uchar>(rowIndex + i);
						for(int j = 0; j < neighbourhood; ++j) {
							sumForMean += p[columnIndex + j];
						}
					}
					double mean = sumForMean / double(neighbourhood * neighbourhood);

					// std of the window
					double sumForStd = 0.0;
					for(int i = 0; i < neighbourhood; ++i) {
						uchar const * p = image.ptr<uchar>(rowIndex + i);
						for(int j = 0; j < neighbourhood; ++j) {
							double t = p[columnIndex + j] - mean;
							sumForStd += t * t;
						}
					}
					double std = sqrt(sumForStd / double(neighbourhood * neighbourhood - 1));

					sumForChoppiness += std;
				}
			}

			return sumForChoppiness / double((image.rows - neighbourhood + 1) * (image.cols - neighbourhood + 1));
		}
	};


	// Base methods for a symmetry
	class Symmetry {
	protected:
		inline double sim(const double a, const double b) const {
			return abs(a - b) < (0.05 * 255.0) ? 1.0 : 0.0;
		}

		double SymmetryHorizontal(const cv::Mat& image) {
			// left vs right
			const double size = image.cols / 2;

			double sum = 0.0;
			for (int i = 0; i < image.rows; ++i) {
				uchar const * p = image.ptr<uchar>(i); // current row
				for (int j = 0; j < size; ++j) {
					sum += sim(p[j], p[image.cols - j - 1]);
				}
			}

			return sum / double(size * image.rows);
		}

		double SymmetryVertical(const cv::Mat& image) {
			// top vs bottom
			const double size = image.rows / 2;

			double sum = 0.0;
			for (int i = 0; i < size; ++i) {
				uchar const * p = image.ptr<uchar>(i); // current row
				uchar const * p2 = image.ptr<uchar>(image.rows - i - 1); // symmetric row
				for (int j = 0; j < image.cols; ++j) {
					sum += sim(p[j], p2[j]);
				}
			}

			return sum / double(size * image.cols);
		}

		double SymmetryDiagonal(const cv::Mat& image) {
			// apply only for a squared subimage
			int startRow = 0;
			int startColumn = 0;
			int lastRow = image.rows - 1;
			int lastColumn = image.cols - 1;
			int size = image.rows;

			if(image.rows != image.cols) {
				size = std::min(image.rows, image.cols);
				startRow = (image.rows - size) / 2;
				startColumn = (image.cols - size) / 2;
				lastRow = startRow + size - 1;
				lastColumn = startColumn + size - 1;
			}

			int halfSize = size / 2;

			double sum = 0.0;
			for(int i = 0; i < halfSize; ++i) {
				for(int j = 0; j < halfSize; ++j) {
					// top left vs botom right
					sum += sim(image.at<uchar>(startRow + i, startColumn + j), image.at<uchar>(lastRow - j, lastColumn - i));
					// bottom left vs top right
					sum += sim(image.at<uchar>(startRow + halfSize + i, startColumn + j), image.at<uchar>(startRow + j, startColumn + halfSize + i));
				}
			}

			return sum / double(2.0 * halfSize * halfSize);
		}
	};

	// Strict symmetry
	struct StrictSymmetry : public Symmetry, public ComponentHelper<FeatureFunction> {
		virtual double Evaluate(const cv::Mat& image) {
			return (SymmetryHorizontal(image) + SymmetryVertical(image) + SymmetryDiagonal(image)) / 3.0;
		}
	};

	// Relaxed symmetry
	struct RelaxedSymmetry : public StrictSymmetry {
		virtual double Evaluate(const cv::Mat& image) {
			const double ss = StrictSymmetry::Evaluate(image) - 0.8;
			return exp(-(ss * ss) / 0.08);
		}
	};


	// Global contrast factor
	class GlobalContrastFactor : public ComponentHelper<FeatureFunction> {
		cv::Mat resizedImages[6];

		inline double PerceptualLuminance(const double k) const {
			return 100.0 * sqrt(pow(k / 255.0, 2.2));
		}

		double Contrast(const cv::Mat& image) {
			const int fullSize = image.rows * image.cols;
			uchar const * p = image.data;

			double sum = 0.0;
			for (int i = 0; i < fullSize; ++i) {
				const double currentPixel = PerceptualLuminance(p[i]);

				double localContrast = 0.0;
				int pixels = 0;

				// left pixel
				if (i - 1 >= 0) {
					localContrast += abs(currentPixel - PerceptualLuminance(p[i - 1]));
					++pixels;
				}
				// right pixel
				if (i + 1 < fullSize) {
					localContrast += abs(currentPixel - PerceptualLuminance(p[i + 1]));
					++pixels;
				}
				// top pixel
				if (i - image.cols >= 0) {
					localContrast += abs(currentPixel - PerceptualLuminance(p[i - image.cols]));
					++pixels;
				}
				// bottom pixel
				if (i + image.cols < fullSize) { 
					localContrast += abs(currentPixel - PerceptualLuminance(p[i + image.cols]));
					++pixels;
				}

				sum += localContrast / double(pixels);
			}

			return sum / double(fullSize);
		}

		inline double WeightForContrast(const double i) const {
			return (-0.406385 * i / 7.0 + 0.334573) * i / 7.0 + 0.0877526;
		}

	public:
		virtual double Evaluate(const cv::Mat& image) {
			// contrast of the original image
			double sum = Contrast(image) * WeightForContrast(1);

			const double contrastScaling[] = { 1.0 / 2.0 , 1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 25.0, 1.0 / 50.0 };

			for (int i = 2; i <= 7; ++i) {
				// scaling factor
				const double scale = contrastScaling[i - 2];
				if(scale * image.cols >= 2 && scale * image.rows >= 2) {
					// resize image
					cv::resize(image, resizedImages[i - 2], cv::Size(), scale, scale, cv::INTER_LINEAR);
					// contrast of the resized image
					sum += Contrast(resizedImages[i - 2]) * WeightForContrast(i);
				}
			}

			return sum;
		}
	};


	// Image complexity by JPEG compression
	class JpegImageComplexity : public ComponentHelper<FeatureFunction> {
		std::vector<uchar> compressedImage;
		cv::Mat decompressedImage;
		std::vector<int> params;

	public:
		JpegImageComplexity() : params(2) {
			params[0] = cv::IMWRITE_JPEG_QUALITY;
			params[1] = 75;
		}

		virtual double Evaluate(const cv::Mat& image) {
			// compress image by JPEG
			cv::imencode(".jpg", image, compressedImage, params);
			// decompress image
			cv::imdecode(compressedImage, cv::IMREAD_GRAYSCALE, &decompressedImage);

			// compression ratio
			const double compressionRatio = double(image.rows * image.cols) / double(compressedImage.size());

			// root mean square between the original and image after compression
			double sum = 0.0;
			for (int i = 0; i < image.cols * image.rows; ++i) {
				double t = image.data[i] - decompressedImage.data[i];
				sum += t * t;
			}
			const double rms = sqrt(sum / double(image.cols * image.rows));

			return rms / compressionRatio;
		}
	};


	// Distance in the pixel space for the given image
	struct DistanceInPixelSpace : public ComponentHelper<FeatureFunction> {
		cv::Mat OriginalImage;

		DistanceInPixelSpace(const cv::Mat& image) : OriginalImage(image.clone()) {}

		virtual double Evaluate(const cv::Mat& image) {
			const double distance = cv::norm(image, OriginalImage, cv::NORM_L2);
			return (distance > 1) ? (1.0 / distance) : 1.0; 
		}
	};


	// Returns constant value
	struct Constant : public ComponentHelper<FeatureFunction> {
		virtual double Evaluate(const cv::Mat& image) {
			return 1.0;
		}
	};



	/// PENALTIES FEATURES (negative features)

	struct JpegImageComplexityPenalty : public JpegImageComplexity {
		virtual double Evaluate(const cv::Mat& image) {
			const double jpegImageComplexity = JpegImageComplexity::Evaluate(image);
			return (jpegImageComplexity > 1) ? 1.0 / jpegImageComplexity : 1.0;
		}
	};

	struct ChoppinessPenalty : public Choppiness {
		virtual double Evaluate(const cv::Mat& image) {
			const double choppiness = Choppiness::Evaluate(image);
			return (choppiness >= 60) ? 1.0 / choppiness : 1.0;
		}
	};

}

#endif	/* FEATURES_HPP */
