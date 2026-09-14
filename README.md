# Former-DFE

This repository is an **extension / modification** of the official implementation of:

> **Former-DFER: Dynamic Facial Expression Recognition Transformer**  
> _Zengqun Zhao, Qingshan Liu_ (ACM MM 2021)  
> Original Repository: [ZengqunZhao/Former-DFER](https://github.com/ZengqunZhao/Former-DFER)

## My modifications and changes:

Add evaluation script for Former-DFER model on DFEW dataset

- Implemented `test_single_clip.py` for evaluating model performance.
- Added argument parsing for model weights and dataset selection.
- Integrated accuracy calculation for top-1 and top-5 metrics.
- Set up data loading for test dataset with specified batch size and workers.
- Included model loading from checkpoint with state dictionary handling.
- Provided detailed logging of evaluation progress and final accuracy.
- Implemented `eval_dfew_all_folds.py` for evaluating the overal model's performance across 5 folds/sets.

<br>

_Zengqun Zhao, Qingshan Liu. "[Former-DFER: Dynamic Facial Expression Recognition Transformer](https://drive.google.com/file/d/12vyWD4mJ9HCkLyBctoPcvUbOU36Ptgc8/view?usp=sharing)". ACM International Conference on Multimedia._

## Setup

`conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch`

## Training on DFEW

- Step 1: download [DFEW](https://dfew-dataset.github.io) dataset.
- Step 2: fill in all the **_your_dataset_path_** in `script.py`, then run `script.py`.
- Step 3: run `sh main_DFEW_trainer.sh`

## Recent Updates

#### Pretrain Models on DFEW

The trained models on DFER (fd1, fd2, fd3, fd4, fd5) can be downloaded [here](https://drive.google.com/drive/folders/1g_n3HURQyQ-oBN6tYvhwD2M32Tdm_9gu?usp=sharing) (Google Driver).

#### Performance on FERV39k

Recently, a new dynamic FER dataset named [FERV39k](https://wangyanckxx.github.io/Proj_CVPR2022_FERV39k.html) is proposed, the results of the Former-DFER on FERV39k are as follows:

| Happiness | Sadness | Neutral | Anger | Surprise | Disgust | Fear |  **UAR**  |  **WAR**  |
| :-------: | :-----: | :-----: | :---: | :------: | :-----: | :--: | :-------: | :-------: |
|   67.57   |  44.16  |  51.81  | 48.93 |  25.09   |  10.80  | 9.80 | **36.88** | **45.72** |

<!-- ## Pre-trained Models

The pre-trained Former-DFER model on DFEW can be downloaded [here](https://drive.google.com/file/d/1YV-KpdYQVAvSQw1setzBF1LeT4qx1bVt/view?usp=sharing). -->

## Citation

If you find our work useful, please consider citing our paper:

```
@inproceedings{zhao2021former,
  title={Former-DFER: Dynamic Facial Expression Recognition Transformer},
  author={Zhao, Zengqun and Liu, Qingshan},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={1553--1561},
  year={2021}
}
```
