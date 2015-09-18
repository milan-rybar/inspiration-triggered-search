# Inspiration-triggered search: Towards higher complexities by mimicking creative processes
Source codes for <http://milanrybar.cz/inspiration-triggered-search/>

Repository does not contain training data set and data of aggregated results.



####Application
The application is implemented in C++11 and used as a library from Python.

Needed libraries to compile the application:
- OpenCV, http://opencv.org/, at least version 3.*
- Boost, http://www.boost.org/
- Python
- Numpy for Python
- PyBrain, http://pybrain.org/
- scikit-learn, http://scikit-learn.org/

Use script 'build.sh' to compile the application.
Python scripts assume that the application has been compiled this way.

Each file with an extension *.dat was saved by Python cPickle library.
See Python scripts for using the application and the meaning of all data from our experiments.
