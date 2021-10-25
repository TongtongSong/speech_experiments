# Conv_Tasnet
语音信号处理秋季本科生课程2020 语音增强部分代码

### Requirement 
- librosa
- pytorch
### 数据准备

数据已经打包到onedrive https://1drv.ms/u/s!AtdlrbUuaZUHm2R1yJQkgKoCEIBg?e=SFOyCj
```shell
python3 createDataPath.py 
```
### Training
- **支持GPU训练**
- **支持模型保存和读取**

由于数据有限 模型训练过程中测试集和验证集保持相同
```shell
python3 main.py 
```
### test
```shell
python3 separate.py 
```

