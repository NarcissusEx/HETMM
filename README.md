# HETMM

The source code of our paper "*[Hard-normal Example-aware Template Mutual Matching for Industrial Anomaly Detection]()*", [Zixuan Chen](https://narcissusex.github.io), [Xiaohua Xie](https://cse.sysu.edu.cn/content/2478), [Lingxiao Yang](https://zjjconan.github.io/), [Jian-Huang Lai](https://cse.sysu.edu.cn/content/2498), *International Journal of Computer Vision (**IJCV**)*, 2024.

**TL;DR:** HETMM is a simple yet effective framework for industrial anomaly detection based on template matching, which can accurately detect and locate unknown anomalies in a training-free manner.

---
## Motivation
<div align=center>
<img width="1148" alt="framework" src="assets/hetmm.png">
</div>

<b>Visualization of training data (ball) and queries (cube) via t-SNE</b>. Visually, existing methods' decision boundaries are dominated by the overwhelming number of easy-normal examples (<b><font style="color:#3288A0">blue balls</font></b>). Hence, the normal queries (<b><font style="color:#6DA945">green cubes</font></b>) near the hard-normal examples (<b><font style="color:#EB720E">orange balls</font></b>) are prone to be erroneously identified as anomalies (<b><font style="color:#B477FD">purple cubes</font></b>), resulting in a high false-positive or missed-detection rate. To address this issue, we propose <b>HETMM</b> to construct a robust prototype-based decision boundary, which can accurately distinguish hard-normal examples from anomalies.

## Framework
<div align=center>
<img width="1148" alt="framework" src="assets/framework.png">
</div>

<b>The overall framework of our methods.</b> In <b>stage I</b>, the original template set $\mathcal{T}^{(j)*}$ is the aggregation of the features extracted by feeding $N$ collected normal images $\mathcal{Z}$ into the pre-trained backbone $\Phi$ with $M$ layers, where each color on $\mathcal{T}^{(j)*}$ denotes that the feature is extracted from different normal images. To streamline $\mathcal{T}^{(j)*}$ into a tiny set $\mathcal{T}^{(j)K}$ with $K$ sheets ($N\ge K$), <b>PTS</b> selects $K$ significant prototypes from $\mathcal{T}^{(j)*}$ at each pixel coordinate through the sliding windows. In <b>stage II</b>, given a query image $q$, we first extract its features $\{Q^{(j)}\}^M _{j=1}$ by the same pre-trained backbone $\Phi$ and then employ <b>ATMM</b> to obtain hierarchical anomaly maps $S^{(j)}$, where each $S^{(j)}$ is generated at the $j$-th layer. $S^\dagger$ is obtained as the final outputs.

## Code Usage
### 1) Get start

* Python 3.9.x
* CUDA 11.1 or *higher*
* NVIDIA RTX 3090
* Torch 1.8.0 or *higher*

**Create a python env using conda**
```bash
conda create -n hetmm python=3.9 -y
```

**Install the required libraries**
```bash
bash setup.sh
```

### 2) Detect and Localize Anomalies on MVTec AD
Using the original template:
```bash
python run.py --ttype ALL --dataset MVTec_AD
```
Using the tiny set formed by PTS (60 sheets):
```bash
python run.py --ttype PTS --tsize 60 --dataset MVTec_AD
```

## Citation

```tex
@article{Chen_2024_hetmm,
    author    = {Chen, Zixuan and Xie, Xiaohua and Yang, Lingxiao and Lai, Jianhuang},
    title     = {Hard-normal Example-aware Template Mutual Matching for Industrial Anomaly Detection},
    journal   = {International Journal of Computer Vision (IJCV)},
    publisher = {Springer}
}
```