A pytorch implement for WWW 2020 paper ''[Leveraging Passage-level Cumulative Gain for Document Ranking](http://www.thuir.cn/group/~YQLiu/publications/WWW2020Wu.pdf)'', namely `Passage-level Cumulative Gain Model (PCGM)`.

## Requirements
* Python 2.7
* Pytorch 0.4.1

## How to use

An example of using the model:

```python
gain = [[0 for _ in range(5)]+[1 for _ in range(5)]+[2 for _ in range(5)]+[3 for _ in range(5)] for j in range(3)]
data = {'passages_id': [[i for i in range(20)] for j in range(3)],
        'gain': gain, # passage-level cumulative gain label
        'pre_gain': [[0]+gain[j][:len(gain[j])-1] for j in range(3)]}

my_model = PCGM()
test_tag = 0 # use gain label of the previous passage as the input of current passage
mask_flag = 1 # enable gain mask
pred = my_model(data, test_tag, mask_flag)
print pred

``` 

You can use the [TianGong-PDR](http://www.thuir.cn/data-pdr/) dataset that contains the PCG labels to train the PCGM. If you have any problems, please contact me via `wuzhijing.joyce@gmail.com`.

## Citation

If you use PCGM in your research, please add the following bibtex citation in your references.

```
@inproceedings{wu2020leveraging,
    title={Leveraging passage-level cumulative gain for document ranking},
    author={Wu, Zhijing and Mao, Jiaxin and Liu, Yiqun and Zhan, Jingtao and Zheng, Yukun and Zhang, Min and Ma, Shaoping},
    booktitle={Proceedings of The Web Conference 2020},
    pages={2421--2431},
    year={2020}
}
```

