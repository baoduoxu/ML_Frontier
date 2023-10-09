# Machine Learning Frontier Project

## Method Chosen

五个数据集包括一共可分为两类:

-  时间序列数据 $$(\{\mathbf{x}^{(1)}_ {t}\} _ {t=1}^T,\cdots,\{\mathbf{x}^{(N)}_ {t}\} _ {t=1}^T)\in \mathbb{R}^{N\times m\times T},\mathbf{x}_{t}^{(i)}\in\mathbb{R}^m$$
- 非时间序列数据 $(\mathbf{x}^{(1)},\cdots,\mathbf{x}^{(N)})\in \mathbb{R}^{N\times m}$

### Method 1 : 对非时序数据处理

对于非时间序列序列的样本考虑降维, 经过试验, 几乎所有的无监督降维方式(包括PCA, LLE, t-SNE, Laplacian, Isomap等)的降维效果都不显著(降维后不成明显的簇), 而 LDA 效果显著, 因此决定用 LDA 来降维.

但是由于 LDA 需要标签, 因此测试集不能使用 LDA 进行降维, 一个朴素的想法是把在训练集和测试集是从同一个分布中进行采样的前提下, 将训练集降维的投影矩阵作用在测试集上, 得到的降维结果与训练集应该是类似的. 不过结果并不是这样, 经过若干次实验, 发现了在小样本的前提下, 对多此采样得到的不同结果使用 LDA 降维得到的结果的分布**完全不一致**, 如下图所示:

![](https://raw.githubusercontent.com/baoduoxu/BlogImage/main/image/202310071908998.jpg)

这九张图, 每一行是不同的随机种子下对数据集进行随机划分的降维结果, 从左到右的图片含义分别为:

1. 使用 LDA 对训练集+验证集进行降维, 数据量为总数据的 80%
2. 使用LDA 对验证集进行降维, 数据量为总数据的 20%
3. 将 1 中降维的投影矩阵作用在测试集上的降维结果, 标签为真实的标签

可以看到训练集的投影矩阵作用在测试集上很不理想，我们推测是样本个数太少导致的，而对于没有标签的测试集，由于所有的无监督降维方式表现也不好，所以我们现在不知道该怎么对测试机降维。





采用降维+SVM的方法, 对ADNI和PPMI的处理方法如下:

  - ADNI. ADNI 需要五分类, 首先直接使用 LDA 降维, 由于降维后 [`MCI`, `MCIn`, `MCIp`] 杂糅在一起而另外两类簇相对独立, 因此对这三类学习一个新的分类器, 称为**子分类器**, 而对所有数据降维操作并使用 SVM 得到的结果为**主分类器**. 对于测试数据, 先通过主分类器进行分类, 如果类别为 `AD` 或者 `NC`, 那么分类错误的概率较小, 直接给其打上相应标签; 如果类别为剩余三类, 就用子分类器再做一次分类.

    设测试集数目为 $N_t$, 验证集样本数目为 $N_{\mathrm{valid}}$,  

    降维从 $\mathbf{x}\in\mathbb{R}^{186}$ 降到 $\hat{\mathbf{x}}\in \mathbb{R}^d$, 学习一个线性变换 $\mathbf{W}\in\mathbb{R}^{d\times d}$ 和偏置 $\mathbf{b}\in\mathbb{R}^{d}$, 设 SVM 得到的超平面为 $\mathbf{P}\in\mathbb{R}^{C\times d},\mathbf{B}\in\mathbb{R}^{C}$, 分别为系数形成的矩阵和截距形成的向量, 经过线性变换和偏置 $\mathbf{Wx}+\mathbf{b}$ 后, 超平面变为 $\mathbf{PW},\mathbf{Pb}+\mathbf{B}$, 于是损失函数为测试集与验证集上超平面的相似程度
    $$
    f(\mathbf{W},\mathbf{b})=\left\Vert{\mathbf{P}_{\mathrm{train}}\mathbf{W}-\mathbf{P}_{\mathrm{valid}}}\right\Vert_F+\left\Vert{\mathbf{P_{\mathrm{train}}b}+\mathbf{B}_{\mathrm{train}}}-\mathbf{B}_{\mathrm{valid}}\right\Vert_2
    $$
    或者误分类率.



- PPMI

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

