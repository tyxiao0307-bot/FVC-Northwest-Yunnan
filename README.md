# Vegetation Cover Analysis in Northwest Yunnan

## Computer Code Availability
The source code for BFAST (Python) and OPGD (R) analyses is permanently archived on GitHub:  
[https://github.com/yourusername/FVC-Northwest-Yunnan](https://github.com/tyxiao0307/FVC-Northwest-Yunnan)  
Anonymous download link: [https://github.com/tyxiao0307/FVC-Northwest-Yunnan/archive/main.zip](https://github.com/tyxiao0307/FVC-Northwest-Yunnan/archive/main.zip)

## What is this repository for?
This repository provides the computational tools for reproducing the analytical results of the study:  
**"Spatial-temporal Variations of Vegetation Cover and Driving Mechanisms in Northwest Yunnan Province, China"**.  

The code enables:
1. Detection of vegetation cover changes using BFAST algorithm
2. Analysis of driving factors with OPGD geodetector
3. Visualization of spatiotemporal patterns

## Quick Test Guide
### Testing BFAST (Python)
```bash
# Install dependencies
pip install numpy scipy statsmodels

# Run test
python test/test_bfast.py

# Expected output:
# Detected breakpoints: [50]
# Magnitude of change: [3.12]
