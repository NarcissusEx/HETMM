# Hard-normal Example-aware Template Mutual Matching for Industrial Anomaly Detection

[Zixuan Chen](https://narcissusex.github.io), [Xiaohua Xie](https://cse.sysu.edu.cn/content/2478), [Lingxiao Yang](https://zjjconan.github.io/), [Jian-Huang Lai](https://cse.sysu.edu.cn/content/2498),

*International Journal of Computer Vision (**IJCV**)*, 2024.

**TL;DR:** HETMM is a simple yet effective framework for industrial anomaly detection based on template matching, which can accurately detect and locate unknown anomalies in a training-free manner.

---
[![Paper](https://img.shields.io/badge/Paper-1?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAADIAAAAuCAYAAABqK0pRAAAKAElEQVRo3u1aWW8b1xlNkfahQG0nLlwg6IIW%2FQnuH%2Bjv6PbsN%2F8Bo08BGsBLU%2BexNYq2iW35wW%2FKQ2HXjrWLnBmSM9xFiqS4WItJStRCUvLXc747Q116QUPJdRIgA1zM8M72Leec77sjvfPOt9u329drE5Fzg8HgV%2F3%2B8MN%2Bv%2F%2Fv%2Ff39eYy5%2F8c4ODiwfh%2Fo7z28r1Zb%2BytM%2Be6Xsfc7luHvDYfDD4%2BOjq5hf%2FXw8DCG47685Q3vxHguOzs9SSSTPkz7fmTgu3t7ez%2BDt7%2FgwO%2Bfc48bPsNYhsFL4Ui8%2BNDnz58L5k88aBSfYY9er6fz0dYfDGRzc2v0e2dnB9fsysbGJh1xRo7gxg9wvoybe7zOGq%2FcXnzxacb29rbu7Y1Gn8iRzc3NH%2FP%2Btw0ROvDs2TMejM0D99J%2B1j6xI%2BtfhSPl1ob0Bodj81P37slfbt4cZeVr7whckblSUxIbu2Ozt2%2FfkWvXr785R%2FggyKoOyKxAqWSAYwiBQP50zj7Pee55jse8fjgYKqlf50i60pBGbzA2e%2BfuXblx48abc6SHG2afzMjMkycyPzcnqURCEp4rszOz4jqOzM%2FOytLCoizML0iQ8vV4eXFR3HhcFjE3PzsnM198Ifl84bU5WcikpLHbfQlaN79Z0BL529ID%2BUdyRoZHx1m7OzUlf%2F7442%2BOIyR7o96QDhTKlltm5DqhFUKSMK2v1U%2FvCHnRB%2B6PDo%2F%2Bp2GRlNq1wi56L17f7nRees7tO3fk6rVrI%2Bd2d3elXF49vSOrpZL4yaS0251Xnl%2BrVCTjB7JSKEoRfPATSSkXV%2FS4kM3inC%2FZIC2ZVEqviQykI81mU3a622OCQGjd%2FOST0XV7J3GE0acRNRjXQDrXKlVZq%2FJ4DQ6VcVzDuVWkeg37qipTEca6yzGIQVISjivxpWXdJ3kM8jtLS%2BJ7niQgApynskUbBcTBvft7%2B6O5z27fHpPfvd09qeBdEznSqtf1hV4sLkEihegmNMJpRFMNiTuS8hL6O5%2FJquOrKyuYdzHnm%2FuSqdE9Pq51YzEp5PL6LF7T7x874uJ5HgaNHWXk7pRcvXp1DFrVSm0yR%2Bq1mr5MjcILGEkaQ1h4MMiDMw4i7iGKNHYXDywXCuJizofjSc2EI0nXU7glHU%2BfxWuZNTrJLEZbNpORJghv8%2Bdfn34qf%2Froo9NxhHgPwAcaTQOyQaCZyKbTmgkaWsjlJIffHP0Dk5GU6yqEUq7JVnxxCRkl1Bx9Du%2F1vaQeR3yg8a3WU9lB43hkcWR6%2BnOZmro3cgTrD2k0mpM50mBGksYADsLHXV4OjzNKWjpAmKQQdVbxarkM%2BMTVSe7TIHcSjtFwOkg4MosRLG2O%2BCykKJxs3aNtfX1jTEz2wB82khM7QgMZOcKDkGBklSswIpdJq8E0iJlje1IpGUey6axmw0DKHTnHZzCLhFALKmXXjET4Lps3efCpZpFboVWaEFp1qJLiGlhOqSMGTjSOGSHMdA6RJuzoSHtrSyNeTAdw0FO5zYSw5D3kBWW4BOmtAuuHwxBaGGlcYxzpjzmSSQWj7h6LPqlV1yYle1XhQJJn8JIgzIiSV7HuKf7VIWTlAPhdyefVEc57IXx4Xnmj1zo6oiwPwuiTIzSaQWP1HsENmc4AwpEAnKiO1KtVzURGlSulL1dsY87IakKdoaMGWgMpg%2BxUJAMjUxeU4K5nnMB81k9rQNLIrF38kuBRFJBoq6MEUACOObI3ufyy8KliwWhGj8ZEXKGRAWuIws5VR0n2Cio%2FocdhMrYMfjh6DdXO1ww5%2BgwWTZsPHrkEx21HSHY%2Bd8yRam3SgtiQfDqjLye%2BTVEzhEzhpSx4EUx4jvJL3LOuqDqxwuNeXsvaQ3hqMPCbGaWzdh3xwEfCct9yJAc%2B8doIWiciexMZSVCBgFHKrk3eSL0IKdYLSvMBsM3ao7DyonO4B4YUcZ7ZZIdAyWam%2BZzh8BhaTsw4b2eAvEkBcnZGVlcrkzmytbGh%2FVUTOGWzuI6CRXIy8uyAo9VgtGfUWEeYDWbL0Qqf1AxFnGLFZ5YiHkV1hPcWoGQ18HJgwS0DGSfZTwUtpp03RuPF1vtVWylsURj9hGsIz8hTxglPSjWdYj9Gp2xHctmc8shWLTaSrEnHqoWMlCfMyHa3i4hktOJS4zuddrj2HowGDTkM51jcWB9IbsKREWd2suACRYE84lzUv%2BXw7HGOuApFmyM%2BeZk45ghhN3GLQtnLM7UwpLxS0pcXtUD5JqqOUSIaSaMYyRWsO5Jh8eR50zUftzmEnPZZoRLyg8TIEa0x7hi0suAW1y12QaxWq5PKb00jSkciY0hsEl4rfbiucMKeSqMFgXAoDFo44UgqoU6bOpRSkWAg3OW4NBuNsRblVfL74pfGE5E9Wu0Fox7LqBCjTwXjcQERY3SDsCK3kHYaQ5Wj5JYKeYVbLmw%2FcuzBoGY53Ef42dBiQUxq89m3CmJDv%2BCMye%2BkHKECcYWXCfFuSOtqtKNGUesB5ukwi1sphFYqXIOktV4QkjnTTAbpkeO6HhnYjiSkgjpkd8ROLBbWkTAj%2B7tSqa5OLr%2BmFrhh45cO60BGI7%2BJmzfW1%2FW6TexpQLlY1FWgj5dzT55wAWZqUUyW5xcNb3RlGYyp1tOnL38qoMgwaFFGuAxm2zKxI37Y9JGkhAOrNAtaNmwiWSuYJf7mS7iWZ8T9MFtOWP1NXYlJsZCRUhXdby0nZYzBcGAtrFq6yrRlnv0Yl8a2%2FJ6oaYzW6TTGtBaGLwotQi0eD9sNX7G9Wlwx58JrymgxVDDYcEIslPi4lo5yXWITmS2MB%2Bm22%2FgglGqbI6snWepG3S4HW3MqD2ESW1jS4YVtOSOuioLud3WlrM2jccCsZeiYp%2BsXo1q694MxPiSpdAiWXUfIRyLBzsjElb0OR%2FywtfZ0TRGuP8IPCrGFRXWKqsTFEslu1vmmnyLEKBaUWha1IAAEs0G4RnFDslvdL%2BbieKb9OaiITqFeq4%2F9fWRjY8JPpr2dbf2uRTiZFZ4pjnkYzcJH%2FLaaLfRgLbxsTSHBjJg236xfeE8a6w8SO0iYjxHRh4tMgMpufXxgxW5vPRuT5An%2FYuW%2B9ksjH8IWhBBgxY1alGiuH35CZTHUvwVGLUx4Pvr7oN43HI4d252v%2FsWq3X5JtWig7dgAx93u8ULLNKuic3HHcd%2FIR%2BzDw%2BGpPmJ3ut2X5jud8TlmMJ8vVre2tgqwtZjNZpcfPPjP32%2FduvXHK1eu%2FBouvBs58hNc3%2F4q%2FqzQbLW6UKUmRosD4tFaWFj45%2BPHj3%2Fz6NGj3z98%2BPAP09PTv7148eIvz549e%2F7MmTM%2FhMlnX%2FmH9Gaz%2BaNer%2Feo0%2Bn43W439bYG33f%2F%2Fv3fXb58%2BYNLly79NBrnz7%2FG0C%2FzTwEXLlz4wblz597HeO8tjvfx7u%2B9qf%2FE%2BC%2FkEJZILUQcyQAAAABJRU5ErkJggg%3D%3D&label=IJCV)](https://link.springer.com/article/10.1007/s11263-024-02323-0)
[![ArXiv](https://img.shields.io/badge/cs.CV-2303.16191-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2303.16191)
[![Google Drive](https://img.shields.io/badge/Material-1?logo=googledrive&label=Google%20Drive)](https://drive.google.com/drive/folders/1c4XvmugX-ryP168bDMFcScdiYWgYktlu?usp=drive_link)
[![Baidu Cloud](https://img.shields.io/badge/Materials-ryP168bDMFcScdiYWgYktlu%3Fusp%3Ddrive_link?logo=Baidu&label=Baidu%20Cloud)](https://pan.baidu.com/s/1HH_3FQo1K72HbUvZpfylxw?pwd=eeg9)

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hard-nominal-example-aware-template-mutual/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=hard-nominal-example-aware-template-mutual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hard-nominal-example-aware-template-mutual/anomaly-detection-on-visa)](https://paperswithcode.com/sota/anomaly-detection-on-visa?p=hard-nominal-example-aware-template-mutual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hard-nominal-example-aware-template-mutual/anomaly-detection-on-surface-defect-saliency)](https://paperswithcode.com/sota/anomaly-detection-on-surface-defect-saliency?p=hard-nominal-example-aware-template-mutual)

## Motivation
<div align=center>
<img width="1148" alt="framework" src="assets/hetmm.png">
</div>

<b>Visualization of training data (ball) and queries (cube) via t-SNE</b>. Visually, existing methods' decision boundaries are dominated by the overwhelming number of easy-normal examples (<b><font style="color:#3288A0">blue balls</font></b>). Hence, the normal queries (<b><font style="color:#6DA945">green cubes</font></b>) near the hard-normal examples (<b><font style="color:#EB720E">orange balls</font></b>) are prone to be erroneously identified as anomalies (<b><font style="color:#B477FD">purple cubes</font></b>), resulting in a high false-positive or missed-detection rate. To address this issue, we propose <b>HETMM</b> to construct a robust prototype-based decision boundary, which can accurately distinguish hard-normal examples from anomalies.

## Framework
<div align=center>
<img width="1148" alt="framework" src="assets/framework.png">
</div>

<b>The overall framework of our methods.</b> In <b>stage I</b>, the original template set $\mathcal{T}^{(j)}$ is the aggregation of the features extracted by feeding $N$ collected normal images $\mathcal{Z}$ into the pre-trained backbone $\Phi$ with $M$ layers, where each color on $\mathcal{T}^{(j)}$ denotes that the feature is extracted from different normal images. To streamline $\mathcal{T}^{(j)}$ into a tiny set $\mathcal{T}^{(j)K}$ with $K$ sheets ($N\ge K$), <b>PTS</b> selects $K$ significant prototypes from $\mathcal{T}^{(j)}$ at each pixel coordinate through the sliding windows. In <b>stage II</b>, given a query image $q$, we first extract its features by the same pre-trained backbone $\Phi$ and then employ <b>ATMM</b> to obtain hierarchical anomaly maps $S^{(j)}$, where each $S^{(j)}$ is generated at the $j$-th layer. $S^\dagger$ is obtained as the final outputs.

## Code Usage
### 1) Get start

* Python 3.9.x
* CUDA 11.1 or *higher*
* NVIDIA RTX 3090
* Torch 1.8.0 or *higher*

**Create a python env using conda**
```bash
conda create -n hetmm python=3.9 -y
conda activate hetmm
```

**Install the required libraries**
```bash
bash setup.sh
```

### 2) Template Generation
**Original template set on MVTec AD:**
```bash
python run.py --mode temp --ttype ALL --dataset MVTec_AD --datapath <data_path>
```
**Tiny set formed by PTS (60 sheets) on MVTec AD:**
```bash
python run.py --mode temp --ttype PTS --tsize 60 --dataset MVTec_AD --datapath <data_path>
```
Since generating pixel-level OPTICS clusters is time-consuming, you can download the "*template*" folder from [Google Drive](https://drive.google.com/drive/folders/1c4XvmugX-ryP168bDMFcScdiYWgYktlu?usp=drive_link) / [Baidu Cloud](https://pan.baidu.com/s/1HH_3FQo1K72HbUvZpfylxw?pwd=eeg9) and copy it into our main folder as:
```
HETMM/
    ├── configs/
    ├── template/
    ├── src/
    ├── run.py
    └── ...
```

### 3) Anomaly Prediction
**Original template set on MVTec AD:**
```bash
python run.py --mode test --ttype ALL --dataset MVTec_AD --datapath <data_path>
```
**Tiny set formed by PTS (60 sheets) on MVTec AD:**
```bash
python run.py --mode test --ttype PTS --tsize 60 --dataset MVTec_AD --datapath <data_path>
```
Please see "*run.sh*" and "*run.py*" for more details.

## Citation

```tex
@article{Chen_2024_hetmm,
    author    = {Chen, Zixuan and Xie, Xiaohua and Yang, Lingxiao and Lai, Jianhuang},
    title     = {Hard-normal Example-aware Template Mutual Matching for Industrial Anomaly Detection},
    journal   = {International Journal of Computer Vision (IJCV)},
    publisher = {Springer},
    year      = {2024},
    doi       = {10.1007/s11263-024-02323-0},
}
```