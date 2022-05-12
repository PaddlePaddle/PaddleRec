mkdir big_train
mkdir big_test
wget -O model_input.tar.gz https://bj.bcebos.com/v1/ai-studio-online/53e61a9bcfc54e0581044883d0f876d9841cb4d0a68848f1a1d568a84591da6f?responseContentDisposition=attachment%3B%20filename%3Dmodel_input.tar.gz&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-04-21T01%3A43%3A00Z%2F-1%2F%2F665a728726f0569e1ef9dd423adfa40a2a5e798f86a8d5d68804a2f21cc03624
tar -zxvf model_input.tar.gz
mv model_input/test_feat_input.pkl big_test/
mv model_input/test_label.pkl big_test/
mv model_input/test_sess_input.pkl big_test/
mv model_input/test_session_length.pkl big_test/
mv model_input/train_feat_input.pkl big_train/
mv model_input/train_label.pkl big_train/
mv model_input/train_sess_input.pkl big_train/
mv model_input/train_session_length.pkl big_train/
