# BMF-SegNet


![Fig1](https://github.com/user-attachments/assets/4ceb4592-2310-45b0-a88a-539c2cd04202)
![2](https://github.com/user-attachments/assets/043f3a1f-f574-4667-acce-4e705509863b)



# Environment install
```python 
git clone https://github.com/hyb2840/BMF-SegNet.git
cd BMF-SegNet
```
# Install mamba
```python 
cd mamba
python setup.py install
```
# Install monai
```python 
pip install monai
```
# train
```python
CUDA_VISIBLE_DEVICES=1 python train.py --arch BMFSegNet --dataset BUS --input_w 128 --input_h 128 --name BUS_BMFSegNet  --data_dir .../inputs/
```
# test
```python
python test.py --name BUS_BMFSegNet --output_dir .../outputs/
```
