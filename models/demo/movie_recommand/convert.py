from paddle_serving_client.io import inference_model_to_serving
inference_model_dir = "user_vector_model/3/"
dir = "user_vector"
serving_client_dir = "{}/serving_client_dir".format(dir)
serving_server_dir = "{}/serving_server_dir".format(dir)
feed_var_names, fetch_var_names = inference_model_to_serving(
		inference_model_dir, serving_server_dir, serving_client_dir)
inference_model_dir = "movie_vector_model/0/"
dir = "movie_vector"
serving_client_dir = "{}/serving_client_dir".format(dir)
serving_server_dir = "{}/serving_server_dir".format(dir)
feed_var_names, fetch_var_names = inference_model_to_serving(
                inference_model_dir, serving_server_dir, serving_client_dir)
