###
# knn - value
###
python downstream_eval.py --target_label value --model_path "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1
python downstream_eval.py --target_label value --model_path "sentence-transformers/roberta-base-nli-stsb-mean-tokens" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1
python downstream_eval.py --target_label value --model_path "sentence-transformers/stsb-bert-base" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1
python downstream_eval.py --target_label value --model_path "sentence-transformers/average_word_embeddings_glove.6B.300d" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1

python downstream_eval.py --target_label value --model_path "google-bert/bert-base-uncased" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50
python downstream_eval.py --target_label value --model_path "FacebookAI/roberta-base" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50
python downstream_eval.py --target_label value --model_path "FacebookAI/xlm-roberta-base" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50

python downstream_eval.py --target_label value --model_path "sentence-transformers/all-mpnet-base-v2" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50
python downstream_eval.py --target_label value --model_path "sentence-transformers/LaBSE" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50
python downstream_eval.py --target_label value --model_path "nomic-ai/nomic-embed-text-v1" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50

python downstream_eval.py --target_label value --model_path "CAiRE/UniVaR-lambda-80" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1
python downstream_eval.py --target_label value --model_path "CAiRE/UniVaR-lambda-20" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1
python downstream_eval.py --target_label value --model_path "CAiRE/UniVaR-lambda-5" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1
python downstream_eval.py --target_label value --model_path "CAiRE/UniVaR-lambda-1" --training_type knn --ouput_dir ./knn --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1

###
# cls - value
###a
python downstream_eval.py --target_label value --model_path "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1
python downstream_eval.py --target_label value --model_path "sentence-transformers/roberta-base-nli-stsb-mean-tokens" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1
python downstream_eval.py --target_label value --model_path "sentence-transformers/stsb-bert-base" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1
python downstream_eval.py --target_label value --model_path "sentence-transformers/average_word_embeddings_glove.6B.300d" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1

python downstream_eval.py --target_label value --model_path "google-bert/bert-base-uncased" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50
python downstream_eval.py --target_label value --model_path "FacebookAI/roberta-base" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50
python downstream_eval.py --target_label value --model_path "FacebookAI/xlm-roberta-base" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50

python downstream_eval.py --target_label value --model_path "sentence-transformers/all-mpnet-base-v2" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50
python downstream_eval.py --target_label value --model_path "sentence-transformers/LaBSE" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50
python downstream_eval.py --target_label value --model_path "nomic-ai/nomic-embed-text-v1" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50

python downstream_eval.py --target_label value --model_path "CAiRE/UniVaR-lambda-80" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1
python downstream_eval.py --target_label value --model_path "CAiRE/UniVaR-lambda-20" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1
python downstream_eval.py --target_label value --model_path "CAiRE/UniVaR-lambda-5" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1
python downstream_eval.py --target_label value --model_path "CAiRE/UniVaR-lambda-1" --training_type cls --ouput_dir ./cls --cache_dir ./cache --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1