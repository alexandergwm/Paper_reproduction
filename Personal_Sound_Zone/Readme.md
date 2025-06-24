
# Some algorithms for personal sound zone simulation

This branch focuses on the simulation of some algorithms for personal sound zone. the basic framework of simulation is based on the paper "Maximization of acoustic energy difference between two spaces". Time-domain ACC and frequency-domain ACC are implemented respectively, namely **ACC**, **AED**, **BACC_RD**, **BACC_RTE**.


## Paper Information

**Algorithm:** ACC (Acoustic Contrast Control)

**Title:** [Robustness and Regularization of Personal  Audio Systems | IEEE TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING](https://ieeexplore.ieee.org/document/6194290/)


**Algorithm:** AED (Acoustic Energy Difference)

**Title:** [Maximization of acoustic energy difference between two spaces | The Journal of the Acoustical Society of America | AIP Publishing](https://pubs.aip.org/asa/jasa/article-abstract/128/1/121/655976/Maximization-of-acoustic-energy-difference-between?redirectedFrom=fulltext)


**Algorithm:** PM-ACC (Pressure Match - Acoustic Contrast Control)

**Title:** [Sound field control with a circular double-layer array of loudspeakers | The Journal of the Acoustical Society of America | AIP Publishing](https://pubs.aip.org/asa/jasa/article-abstract/131/6/4518/656056/Sound-field-control-with-a-circular-double-layer?redirectedFrom=fulltext)

**Algorithm:** BACC_RD (Broadband Acoustic Contrast Control with Response Differential)

**Title:** [Time-domain acoustic contrast control design with response differential constraint in personal audio systems | The Journal of the Acoustical Society of America | AIP Publishing](https://pubs.aip.org/asa/jasa/article/135/6/EL252/607071/Time-domain-acoustic-contrast-control-design-with)

**Algorithm:** BACC_RTE (Broadband Acoustic Contrast Control with Response Trend Estimation)

**Title:** [Time-domain acoustic contrast control design with response differential constraint in personal audio systems | ICASSP ](https://sps.ewi.tudelft.nl/pubs/schellekens16icassp.pdf)

## Usage

The simulation framework provides two main implementations for different array configurations:

### Circular Array Configuration
Use the **`Personal Sound Zone_circular.ipynb`** notebook for circular loudspeaker array simulations. This implementation supports all aforementioned algorithms (ACC, AED, BACC_RD, BACC_RTE) in a circular array geometry.

### Linear Array Configuration  
Use the **`Personal Sound Zone_line.ipynb`** notebook for linear loudspeaker array simulations. This provides similar algorithm implementations optimized for linear array configurations.

### Quick Start
1. Open the desired notebook file in Jupyter environment
2. Configure simulation parameters (array geometry, frequency range, etc.)
3. Select the algorithm (ACC/AED/PM-ACC/BACC_RD/BACC_RTE)
4. Run the cells to execute the simulation and visualize results

Both notebooks include comprehensive examples and parameter configurations for reproducing the results presented in the referenced papers.
