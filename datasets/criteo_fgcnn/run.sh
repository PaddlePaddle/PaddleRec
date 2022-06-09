wget --no-check-certificate https://paddlerec.bj.bcebos.com/datasets/fgcnn/datapro.zip
unzip -o datapro.zip	
echo "Complete data download."
mkdir train
mkdir test
mv criteo_x4_5c863b0f_c15c45a1/train.h5 train
mv criteo_x4_5c863b0f_c15c45a1/valid.h5 test
