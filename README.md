# Reliable Programmatic Weak Supervision with Confidence Intervals for Label Probabilities

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](/AMRC_Python) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](#support-and-author)

This repository is the official implementation of Reliable Programmatic Weak Supervision with Confidence Intervals for Label Probabilities. This paper presents a methodology for programmatic weak supervision that can provide confidence intervals for label probabilities and obtain more reliable predictions. In particular, the methods proposed use uncertainty sets of distributions that encapsulate the information provided by LFs with unrestricted behavior and typology.

This is a fork of the [wrench](https://github.com/JieyuZ2/wrench) repository that focuses on aggregating the provided labeling functions for certain datasets.

## Installation
1. create and activate conda environment WITHOUT using the `environment.yml` file

    `conda create --name wrench python=3.6`

    `conda activate wrench`
2. install wrench

    `pip install ws-benchmark==1.1.2rc0`
3. install other dependencies

    `pip install -r requirements.txt`

## Support and Author

Verónica Álvarez

valvarez@bcamath.org

[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://github.com/VeronicaAlvarez)

## License 

The proposed methods carry a MIT license.

## Citation

If you find useful the code in your research, please include explicit mention of our work in your publication with the following corresponding entry in your bibliography:

<a id="1">[1]</a> 
V. Alvarez, S. Mazuelas, S. An, S. Dasgupta.
Reliable Programmatic Weak Supervision with Confidence Intervals for Label Probabilities





