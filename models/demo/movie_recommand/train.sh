echo "Recall offline training ..."
echo "Model config at models/demo/movie_recommand/recall/config.yaml"
python -m paddlerec.run -m ./recall/config.yaml 

echo "----------------------------------------"
echo "Rank offline training ..."
echo "Model config at models/demo/movie_recommand/rank/config.yaml"
python -m paddlerec.run -m ./rank/config.yaml 
