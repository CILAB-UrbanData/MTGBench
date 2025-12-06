<img src="fig/MTG-logo-hires-transparent.png" width="260" align="left" />
<br clear="left" /><br>

------

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![Pytorch](https://img.shields.io/badge/Pytorch-2.2.1%2B-blue)](https://pytorch.org/) ![Gitea Stars](https://img.shields.io/gitea/stars/CILAB-UrbanData/MTGBench)


# MTGBench

[Dataset Download](https://cilab-urbandata.github.io/) | [Dataset Processing Code](https://github.com/CILAB-UrbanData/MTGBench-Dataset) | [Conference Paper]() | [中文](https://github.com/CILAB-UrbanData/MTGBench/blob/master/README_zh.md)

---

MTGBench is a unified, novel, and extensible benchmarking framework designed for emerging traffic prediction tasks.  
It provides a reliable development and evaluation platform built upon PyTorch, with improvements inspired by excellent open-source frameworks such as [TSLib](https://github.com/thuml/Time-Series-Library/tree/main) and [LibCity](https://github.com/LibCity/Bigscity-LibCity?tab=readme-ov-file).

MTGBench currently supports:

* **Traffic State Prediction**
  * Trajectory-based traffic state prediction  
  * Order-based traffic state prediction  

---

## Features

### **• Unified**
MTGBench provides a systematic pipeline integrating model implementation, usage, and evaluation into a single unified platform. It includes standardized spatiotemporal data formats, a unified model instantiation interface, and consistent evaluation procedures.

### **• Novel**
MTGBench emphasizes emerging traffic prediction tasks, where models may include trajectories or OD flows as inputs in addition to traditional traffic state data.

### **• Extensible**
MTGBench adopts a fully modular design that allows users to flexibly incorporate custom components.  
Researchers can easily develop new models on top of MTGBench.

---

## Overall Framework

<img src="fig/pipeline_original.png" width="500" align="left" />
<br clear="left" /><br>

* **./scripts/** — Shell scripts for launching each task/model with corresponding hyperparameters  
* **./data_provider/** —  
  * `data_factory.py`: instantiates different data_loader objects  
  * `./data_provider/data_loader/`: contains data reading, preprocessing, and sampling logic for each model  
* **./exp/** — Unified training and evaluation scripts across tasks  
* **run.py** — Main program entry: loads hyperparameters, instantiates experiment classes, and executes training and testing  

---

## Quick Start

Before running any model in MTGBench, ensure you have downloaded at least one dataset and placed it under `./data/`.  
Datasets can be downloaded from:  
👉 **https://cilab-urbandata.github.io/**

All datasets must be preprocessed according to the format described in:  
👉 **https://github.com/CILAB-UrbanData/MTGBench-Dataset**

To start training or testing, simply run:

```bash
./scripts/traffic_prediction/sf/TrGNN.sh
```
This script runs the TrGNN model on the San Francisco dataset for traffic state prediction using default configurations.

## Tutorial
To add your own model to MTGBench:

* Add your model file under `./models/`. For reference, see: `./models/TrGNN.py`. Then Register your model in
`./exp/exp_basic.py` → `Exp_Basic.model_dict`

* Register your data processing components in
`./data_provider/data_factory.py` → `data_dict`
and
`./data_provider/data_loader/__init__.py`

* Add a corresponding launch script under `./scripts/`
