# Llama2 Inferencing from Scratch

This repository contains an implementation of Llama2 inferencing from scratch using PyTorch. It includes a requirements file for dependencies, a `model.py` file for the model definition, and an `inference.py` file for running inference. Weights can be downloaded from their official website.



## Installation

To get started, clone the repository and install the required dependencies using pip:

```bash
git clone https://github.com/raahul4004/llama2-from-scratch.git
cd llame2-from-scratch
pip install -r requirements.txt
```


## Model

### Llama2-7B Weights

The model used in this implementation is based on the Llama2 architecture and specifically uses the Llama2-7B weights. These weights are pre-trained and provide a strong starting point for various inferencing tasks.

### Model Definition

The `model.py` file contains the definition of the Llama2 model. You can customize the model architecture based on your requirements.


### Running Inference

To run inference using the pre-defined model, use the `inference.py` script. This script loads the model and performs inference on the provided input data.

```bash
python inference.py 
```

## Files

- `requirements.txt`: Contains the list of dependencies required to run the project.
- `model.py`: Defines the Llama2 model using PyTorch.
- `inference.py`: Script to perform inference using the Llama2 model.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
