# Vehicle Re-id Using Triplet Networks
The objective is to train a Triplet Network for Vehicle Re-Identification using VeRi Dataset

## Getting Started
### Prerequisites
1. Pytorch


### Dataset
The VeRi dataset can be downloaded from the link given below.
 https://drive.google.com/file/d/0B0o1ZxGs_oVZWmtFdXpqTGl3WUU/view
 
 
### Training

```
python trainer.py --use_cuda --gpu 1 --train_images /home/neuroplex/Downloads/VeRi/image_train --test_images /home/neuroplex/Downloads/VeRi/image_train  --train_annotation_path /home/neuroplex/Downloads/VeRi/train_label.xml  --test_annotation_path /home/neuroplex/Downloads/VeRi/test_label.xml
```

### Inference
Download the pretrained weights from -
https://drive.google.com/file/d/12CX-4W5doq6ZE87e3IrDOUv_iIVHaNyX/view?usp=sharing

```
python compare.py
```

## License
This project is licensed under the MIT License 
