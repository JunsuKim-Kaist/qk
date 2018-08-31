# Qualcomm-Kaist Hackerthon

Let's collect lots of open source as possible!

libraries to install are in requirements.txt

preprocess :

python3 makeAlignedFacevideo.py

python3 makeDataset_QIA.py

train :

cd qk/src/

python3 mainpro_qia.py

python3 main_qia.py --pretrained

if you want to start from checkpoint add --resume argument

test:

cd qk/Preprocessing/

python3 makeAlignedFaceVideo_FinalTest.py

python3 makeDataset_FinalTest.py --label_blind

python3 plot_final_confusion_matrix.py --label_blind

then, you can get result.xlsx

