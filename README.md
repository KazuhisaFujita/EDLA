# Error Diffusion Learning Algorithm (EDLA)

This repository contains an implementation of the **Error Diffusion Learning Algorithm (EDLA)**, a biologically-inspired neural network training method originally proposed by Kaneko (Kaneko). EDLA features positive and negative neurons interconnected by excitatory and inhibitory synapses and utilizes a global error diffusion mechanism. By diffusing a single global error throughout the network, EDLA simplifies the learning process.

## Repository Contents

- `EDLA.py`: Core implementation of the EDLA network architecture.
- `EDLA_no_negative_in_the_last_layer.py`: Variant of the EDLA model without negative neurons in the last layer.
- `criterion.py` & `criterion_reg.py`: Evaluation functions for trained neural networks, providing criterion values for classification and regression tasks, respectively.
- `datasets.py` & `data_reg.py`: Custom dataset loaders for classification and regression tasks, respectively.
- `EDLA_digits.ipynb`: Jupyter notebook demonstrating EDLA on image classification tasks (Digits dataset).
- `EDLA_regression.ipynb`: Jupyter notebook demonstrating EDLA on regression tasks.

## References

- Kazuhisa Fujita (2025) arxiv.
- Isam Kaneko. 誤差拡散学習法のサンプルプログラム. https://web.archive.org/web/20000306212433/http://village.infoweb.ne.jp/~fwhz9346/ed.htm.

## License

This project is licensed under the MIT License.

