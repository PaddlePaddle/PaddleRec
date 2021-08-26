wget https://paddlerec.bj.bcebos.com/datasets/omniglot/omniglot_python.zip
unzip omniglot_python.zip
mv images_evaluation/* images_background/
mv images_background omniglot_raw
rm -rf demo.py images_background_small1 images_background_small2 images_evaluation/ one-shot-classification strokes_*
python preprocess.py
