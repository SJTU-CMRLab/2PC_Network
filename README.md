# Joint suppression of cardiac bSSFP cine banding and flow artifacts using twofold phase-cycling and a dual-encoder neural network

![](https://github.com/SJTU-CMRLab/2PC_Network/blob/main/image.png))

## Abstract
**Background** <br />
Cardiac bSSFP cine imaging suffers from banding and flow artifacts induced by off-resonance. Although fourfold phase cycling suppresses banding artifacts, it invokes flow artifacts and prolongs the scan. The purpose of this work was to develop a twofold phase cycling sequence with a neural network-based reconstruction (2PC+Network) for a joint suppression of banding and flow artifacts in cardiac cine imaging. <br />
**Methods** <br />
A dual-encoder neural network was trained on 1620 pairs of phase-cycled left ventricular (LV) cine images from 18 healthy subjects. Twenty healthy subjects and ten patients were prospectively scanned using the proposed 2PC sequence. Regular bSSFP cine, regular cine with a neural network-based artifact reduction (pure post-processing), regular 2PC based on averaging of the two phase-cycled images, and the proposed method were mutually compared, in terms of artifact suppression performance in the LV, generalizability over altered scan parameters and scanners, and the suppression of large-area banding artifacts in the left atrium (LA).   <br />
**Results** <br />
The proposed 2PC+Network showed robust suppressions of artifacts across a range of center frequency offsets. Compared with bSSFP and 2PC, the 2PC+Network improved banding artifacts (3.85±0.67 and 4.50±0.45 vs 5.00±0.00, p<0.01 and p=0.02), flow artifacts (3.35±0.78 and 2.10±0.77 vs 4.90±0.20, both p<0.01), and overall image quality (3.25±0.51 and 2.30±0.60 vs 4.75±0.25, both p<0.01). Post-processing and 2PC+Network achieved a similar artifact suppression performance, yet the latter achieved better authenticity scores (two-chamber, 3.55±0.65 vs 4.60±0.62, p<0.01; four-chamber, 3.15±0.78 vs 3.90±0.73, p=0.02; and LA, 3.60±0.54 vs 4.70±0.33, p<0.01) with fewer hallucinations. Furthermore, in the pulmonary vein ostium and LA, post-processing cannot eliminate banding artifacts since they occupied a large area, whereas 2PC+Network reliably suppressed the artifacts. <br />
**Conclusions** <br />
The proposed 2PC+Network method jointly suppresses banding and flow artifacts while manifesting a good generalizability against variations of anatomy and scan parameters. It provides a robust and practical tool for suppression of the two types of artifacts in cardiac cine imaging. <br />

## Requirements
numpy==1.21.2 <br />
opencv_python==4.5.1.48 <br />
scipy==1.8.1 <br />
torch==1.11.0 <br />
torchvision==0.12.0 <br />
## How to use
Details of the code are as follows:

[train.py](https://github.com/SJTU-CMRLab/2PC_Network/blob/main/train.py): To train the dual-encoder neural network.

[test.py](https://github.com/SJTU-CMRLab/2PC_Network/blob/main/test.py): To test the dual-encoder neural network.

[data](https://github.com/SJTU-CMRLab/2PC_Network/blob/main/data): It contains training data and testing data. Training data consists of bSSFP data with 12 different RF phase increments and corresponding labels.

[model](https://github.com/SJTU-CMRLab/2PC_Network/blob/main/model): It contains trained model, which can be used in the testing process.

## How to cite this work
If you want to cite this work or use part of the code, please cite:  

Joint suppression of cardiac bSSFP cine banding and flow artifacts using twofold phase-cycling and a dual-encoder neural network, Zhuo Chen1#, Yiwen Gong2#, Haiyang Chen1, Yixin Emu1, Juan Gao1, Yiwen Shen2, Xin Tang3, Sha Hua2, Wei Jin2, and Chenxi Hu1, * (under review)
<br />

