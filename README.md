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
@inproceedings{Leveraging2020Wu,
    author = {Wu, Zhijing and Mao, Jiaxin and Liu, Yiqun and Zhan, Jingtao and Zheng, Yukun and Zhang, Min and Ma, Shaoping},
    title = {Leveraging Passage-Level Cumulative Gain for Document Ranking},
    year = {2020},
    isbn = {9781450370233},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3366423.3380305},
    doi = {10.1145/3366423.3380305},
    booktitle = {Proceedings of The Web Conference 2020},
    pages = {2421–2431},
    numpages = {11},
    keywords = {document ranking, Passage-level cumulative gain, neural network},
    location = {Taipei, Taiwan},
    series = {WWW '20}
}
```

