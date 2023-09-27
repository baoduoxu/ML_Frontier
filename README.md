# Machine Learning Frontie Project

## Method Chosen

因此五个数据集包括一共可分为两类:

-  时间序列数据 $(\{\mathbf{x}^{(1)}_{t}\}_{t=1}^T,\cdots,\{\mathbf{x}^{(N)}_{t}\}_{t=1}^T)\in \mathbb{R}^{N\times m\times T},\mathbf{x}_{t}^{(i)}\in\mathbb{R}^m$
- 非时间序列数据 $(\mathbf{x}^{(1)},\cdots,\mathbf{x}^{(N)})\in \mathbb{R}^{N\times m}$

## Description of Datasets

一共是如下的五个数据集, 中括号内为该数据集中所包含的所有标签.

**1.**   **ADNI**： ['AD', 'MCI', 'MCIn', 'MCIp', 'NC']

> $N=51+99+56+43+52,m=186$.

**2.**   **ADNI_90_120_fMRI**：['AD', 'EMCI', 'LMCI', 'NC']

> $N=59+56+43+48,m=90,T=120$.

**3.** **FTD_90_200_fMRI**：['FTD', 'NC']

> $N=95+86,m=90,T=200$.

**4.** **OCD_90_200_fMRI**：['NC', 'OCD']

>$N=95+86,m=90,T=200$.

**5.**   **PPMI**：['NC', 'PD']

> $N=169+374,m=294$.

### Meaning of Datasets and Labels

#### Datasets

**ADNI**：与认知功能有关的神经退行性疾病. 

**FTD**：前额颞叶退行性病变.

**OCD**：强迫症障碍病.

**PPMI**：帕金森性疾病.

#### Labels

**AD**：阿尔茨海默病；

**MCI**：轻度认知功能障碍；

**MCIn**：某些患有MCI的个体在一段时间内未表现出认知功能的明显恶化；

**MCIp**：某些患有MCI的个体在一段时间内经历了认知功能的明显恶化；

**NC**：正常对照组，这是一组没有明显认知功能问题的健康个体；

**EMCI**：早期阶段被诊断为轻度认知功能障碍的个体；

**LMCI**：晚期轻度认知功能障碍，这些个体的认知功能问题可能已经更加明显，并可能是认知障碍的前兆；

**FTD**：前额颞叶退行性病变；

**OCD**：表示强迫症障碍；

**PD**：表示帕金森病.

### 数据集里不同维度解释

1.      ADNI: (e.g., AD的shape为(51, 186)：51表示样本数，186表示特征维度).
2.      ADNI_90_120_fMRI: (e.g., AD的shape为(59, 90, 120)：59表示样本数，90表示脑区数，120表示时间序列).
3.      FTD_90_200_fMRI: (e.g., NC的shape为(86, 90, 200)：86表示样本数，90表示脑区数，200表示时间序列).
4.      OCD_90_200_fMRI: (e.g., NC的shape为(20, 90, 200)：20表示样本数，90表示脑区数，200表示时间序列).
5.      PPMI: (e.g., PD的shape为(374, 294)：374表示样本数，294表示特征维度).

