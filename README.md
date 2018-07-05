<a href="https://travis-ci.org/wilson-eft/wilson">![Build Status](https://travis-ci.org/wilson-eft/wilson.svg?branch=master)</a> [![pipeline status](https://hepcode.net/wilson/wilson/badges/master/pipeline.svg)](https://hepcode.net/wilson/wilson/commits/master) [![Coverage Status](https://coveralls.io/repos/github/wilson-eft/wilson/badge.svg)](https://coveralls.io/github/wilson-eft/wilson) [![coverage report](https://hepcode.net/wilson/wilson/badges/master/coverage.svg)](https://hepcode.net/wilson/wilson/commits/master)

# wilson – running and matching beyond the Standard Model

wilson is a Python package for the running and matching of Wilson coefficients of higher-dimensional operators beyond the Standard Model of particle physics. It implements the one-loop running of all dimension-6 operators in the Standard Model Effective Theory (SMEFT), complete tree-level matching onto the weak effective theory (WET) at the electroweak scale, and complete one-loop running of all dimension-6 WET operators in QCD and QED. It uses the Wilson coefficient exchange format ([WCxf](https://wcxf.github.io)) for representing Wilson coefficient values and can be easily interfaced with all codes supporting this standard.

## Installation

The package requires Python version 3.5 or above. It can be installed with

```bash
python3 -m pip install wilson
```


## Documentation

A brief user manual can be found in the paper cited below. More information can be found on the [project web site](https://wilson.github.io).

## Citation

If you use wilson in a project, please cite:

TBD

## Related work


- The SMEFT RGEs are based on [arXiv:1308.2627](https://arxiv.org/abs/1308.2627), [arXiv:1310.4838](https://arxiv.org/abs/1310.4838), and [arXiv:1312.2014](https://arxiv.org/abs/1312.2014).
- The SMEFT RGE implementation was ported from the [DSixTools](https://dsixtools.github.io/) Mathematica package ([arXiv:1704.04504](https://arxiv.org/abs/1704.04504)).
- The SMEFT to WET matching is based on [arXiv:1709.04486](https://arxiv.org/abs/1709.04486).
- The QCD and QED running is based on [arXiv:1704.06639](https://arxiv.org/abs/1704.06639) and [arXiv:1711.05270](https://arxiv.org/abs/1711.05270).



## Bugs and feature requests

Please submit bugs and feature requests using
[Github's issue system](https://github.com/wilson-eft/wilson/issues).


## Contributors

In alphabetical order:

- Jason Aebischer
- Jacky Kumar
- Xuanyou Pan
- Matthias Schöffel
- David M. Straub

## License

wilson is released under the MIT license.
