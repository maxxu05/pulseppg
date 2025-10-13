# Pulse-PPG: An Open-Source Field-Trained PPG Foundation Model for Wearable Applications Across Lab and Field Settings
Mithun Saha<sup>1,†</sup>, Maxwell A. Xu<sup>2,†</sup>, Wanting Mao<sup>2</sup>, Sameer Neupane<sup>1</sup>, James M. Rehg<sup>2</sup>, Santosh Kumar<sup>1</sup>

<sub><sup>†</sup>Co-first authors &nbsp; &nbsp; | &nbsp; &nbsp; <sup>1</sup>University of Memphis <sup>2</sup>University of Illinois Urbana-Champaign</sub>


####   Accepted at UbiComp, ACM IMWUT, 2025. Please read our paper here: [https://dl.acm.org/doi/abs/10.1145/3749494](https://dl.acm.org/doi/abs/10.1145/3749494).



## Code Overview
Below is an outline of the overall structure of our codebase. The code is nicely modularized with modular class-based configs that help define specific components of an experiment, such as a config for tuning the model training or a config for designing the network backbone. Extending this codebase to your own use-cases should be fairly straightforward.
                    
```
run_exp.py           # Main file used to launch experiments  
pulseppg/            # Source code  
├── experiments/      
│   └── configs/     # Config for defining experiment
├── models/          # Training pipeline
│   └── RelCon/      # RelCon trainer for Pulse-PPG FM
│   └── MotifDist/  
├── nets/            # Network backbones (e.g. ResNet)  
├── data/            
│   └── process/     # Downloading and preprocessing data  
└── eval/            # Evaluation pipeline  
```

## Code Usage

### (A) Download Model Weights

The pre-trained model weights are available on Zenodo at this DOI [10.5281/zenodo.17270930](https://doi.org/10.5281/zenodo.17270930). Here we provide this bash script for your convenience for downloading and unpacking the weights. 

    bash ./download_pulseppg.sh


### (B) Python Environment

For this project we use miniconda to manage dependencies. [After installing miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2), we can install the pulseppg environment with the following terminal commands:

    conda env create -f env.yml
    conda activate pulseppg
    pip install -e . 


## Citation
If you use our work in your research, please cite

```bibtex
@article{saha2025pulse,
  title={Pulse-ppg: An open-source field-trained ppg foundation model for wearable applications across lab and field settings},
  author={Saha, Mithun and Xu, Maxwell A and Mao, Wanting and Neupane, Sameer and Rehg, James M and Kumar, Santosh},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={9},
  number={3},
  pages={1--35},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```
If you have any further questions, please feel free to email me at maxu@illinois.edu
