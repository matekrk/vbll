# Variational Bayesian Last Layers (VBLL)

VBLL introduces a deterministic variational formulation for training Bayesian last layers in neural networks. This method offers a computationally efficient approach to improving uncertainty estimation in deep learning models. By leveraging this technique, VBLL can be trained and evaluated with quadratic complexity in last layer width, making it nearly computationally free to add to standard architectures. Our work focuses on enhancing predictive accuracy, calibration, and out-of-distribution detection over baselines in both regression and classification.

## Installation
The easiest way to install VBLL is with pip:
```bash
pip install vbll
```

You can also install by cloning the GitHub repo:
```bash
# Clone the repository
git clone https://github.com/VectorInstitute/vbll.git

# Navigate into the repository directory
cd vbll

# Install required dependencies
pip install -e .
```

## Usage and Tutorials
Documentation is available [here](https://vbll.readthedocs.io/en/latest/). 

You can also check out our tutorial colabs: 

- Regression: <a href="https://colab.research.google.com/github/VectorInstitute/vbll/blob/main/docs/tutorials/VBLL_Regression.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- Classification: <a href="https://colab.research.google.com/github/VectorInstitute/vbll/blob/main/docs/tutorials/VBLL_Classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Contributing
Contributions to the VBLL project are welcome. If you're interested in contributing, please read the contribution guidelines in the repository.

## Citation
If you find VBLL useful in your research, please consider citing our paper:

```bibtex
@inproceedings{harrison2024vbll,
  title={Variational Bayesian Last Layers},
  author={Harrison, James and Willes, John and Snoek, Jasper},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```


