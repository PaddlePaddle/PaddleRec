mkdir big_train
mkdir big_test
wait1 = `wget -O model_input.tar.gz https://paddlerec.bj.bcebos.com/datasets/Ali_Display_Ad_Click/model_input.tar.gz`
wait2 = `tar -zxvf model_input.tar.gz`
mv model_input/test_feat_input.pkl big_test/
mv model_input/test_label.pkl big_test/
mv model_input/test_sess_input.pkl big_test/
mv model_input/test_session_length.pkl big_test/
mv model_input/train_feat_input.pkl big_train/
mv model_input/train_label.pkl big_train/
mv model_input/train_sess_input.pkl big_train/
mv model_input/train_session_length.pkl big_train/
echo "data prepare done!"
