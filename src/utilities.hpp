#ifndef UTILITIES_HPP
#define	UTILITIES_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <boost/format.hpp>

namespace its {

	template<typename T>
	void Save(const std::vector<T>& data, std::ostream& stream, const char sep = '\t') {
		for (int i = 0; i < data.size(); ++i) {
			if (i != 0) {
				if(sep == 0) stream << std::endl; 
				else stream << sep;
			}
			stream << data[i];
		}
		stream << std::endl;
	}

	template<typename T>
	void Save(const std::vector<std::vector<T>>& data, std::ostream& stream) {
		for (int i = 0; i < data.size(); ++i) {
			Save(data[i], stream);
		}
	}

	template<typename T>
	void SaveAsText(const std::vector<T>& data, const std::string& filename) {
		std::ofstream file(filename);
		Save(data, file, 0);
		file.close();
	}

	template<typename T>
	void SaveAsText(const std::vector<std::vector<T>>& data, const std::string& filename) {
		std::ofstream file(filename);
		Save(data, file);
		file.close();
	}


	template <typename Formater>
	inline void FormatRecursive(Formater& f) {}

	template<typename Formater, typename T, typename... Args>
	inline void FormatRecursive(Formater& f, T t, Args... args)
	{
			f % t;

			FormatRecursive(f, args...) ;
	}

	template<typename... Args>
	std::string Format(const std::string& format, Args... args)
	{
		auto f = boost::format(format); 
		FormatRecursive(f, args...);
		return f.str();
	}

}

#endif	/* UTILITIES_HPP */
