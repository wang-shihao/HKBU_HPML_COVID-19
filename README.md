# HKBU_HPML_COVID-19

## 1. Link to Clean-CC-CCII (Based on CC-CCII Ver. 1.0)
FTP Server: http://47.92.107.188/Datasets/COVID-DATA-CT/

Google Drive: https://drive.google.com/drive/folders/1qOWNdi5eRpuJClPimwIHvCV8z2RN7HQB?usp=sharing

OneDrive: https://hkbuedu-my.sharepoint.com/:f:/g/personal/shwang_link_hkbu_edu_hk/EoieIbNgTjNGh0-sbsx3kj8BxzurNH994kwX2R6mjf3q1g?e=miYpfz

Raw Data (CC-CCII) http://ncov-ai.big.ac.cn/download

## 2. Experimental results

### 2.1 Benchmark Deep Learning Models

> If you want to run the benchmark experiments, you can refer to the directory of `covid19_pipeline`.

- The pipeline of benchmarking deep learning-based models.

![pipeline](./images/pipeline.png)

- Performance comparison between different models

![model_perf](./images/model_perf.png)


- Performance comparison between ResNet3d models with different depth

![model_depth](./images/model_depth.png)



- Performance comparison between models trained by scan data comprising a different number of slices.

![mdoel_slice](./images/mdoel_slice.png)



- The model accuracy before and after using MixUp data augmentation method.

![model_mixup](./images/model_mixup.png)


### 2.2 Automated model design

> The code of NAS will be released very soon ...

- NAS pipeline

![NAS pipeline](./images/nas_pipeline.png)

- Search space

![Search space](./images/search_space.png)

- The performance comparison between baseline models and models designed by NAS


![nas_vs_manual](./images/nas_vs_manual.png)

