## DeepTMpred2

### Dataset
Orientations of Proteins in Membranes (OPM) database: https://opm.phar.umich.edu/download

### Performance comparison of test sets

| Methdos     | F1(r) | F(h)  | V(p) |
| ----------- | ----- | ----- | ---- |
| DeepTMpred  | 0.900 | 0.867 | 28   |
| DeepTMpred2 | 0.904 | 0.874 | 30   |

### TMPs prediction script

```shell script
python predict.py --input test.fa --output test.json &
```

### License
[MIT](LICENSE)

### Contact
If you have any questions, comments, or would like to report a bug, please file a Github issue or 
contact me at wanglei94@hust.edu.cn.
