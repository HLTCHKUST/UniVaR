#######
###
# Text-Only
###
#######

###
# knn - value
###
python translationese_eval.py --model_path "sentence-transformers/average_word_embeddings_glove.6B.300d" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1

python translationese_eval.py --model_path "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1
python translationese_eval.py --model_path "sentence-transformers/roberta-base-nli-stsb-mean-tokens" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1
python translationese_eval.py --model_path "sentence-transformers/stsb-bert-base" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1

python translationese_eval.py --model_path "sentence-transformers/all-mpnet-base-v2" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50
python translationese_eval.py --model_path "sentence-transformers/LaBSE" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50
python translationese_eval.py --model_path "nomic-ai/nomic-embed-text-v1" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50

python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-80" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-20" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-5" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-1" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1

###
# cls - value
###a
python translationese_eval.py --model_path "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1

python translationese_eval.py --model_path "sentence-transformers/roberta-base-nli-stsb-mean-tokens" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1
python translationese_eval.py --model_path "sentence-transformers/stsb-bert-base" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1
python translationese_eval.py --model_path "sentence-transformers/average_word_embeddings_glove.6B.300d" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1

python translationese_eval.py --model_path "sentence-transformers/all-mpnet-base-v2" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50
python translationese_eval.py --model_path "sentence-transformers/LaBSE" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50
python translationese_eval.py --model_path "nomic-ai/nomic-embed-text-v1" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50

python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-80" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-20" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-5" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-1" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1

#######
###
# Translation
###
#######

###
# knn - value
###
python translationese_eval.py --model_path "sentence-transformers/average_word_embeddings_glove.6B.300d" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name translation
python translationese_eval.py --model_path "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name translation
python translationese_eval.py --model_path "sentence-transformers/roberta-base-nli-stsb-mean-tokens" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name translation
python translationese_eval.py --model_path "sentence-transformers/stsb-bert-base" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name translation

python translationese_eval.py --model_path "sentence-transformers/all-mpnet-base-v2" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --task_name translation
python translationese_eval.py --model_path "sentence-transformers/LaBSE" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --task_name translation
python translationese_eval.py --model_path "nomic-ai/nomic-embed-text-v1" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --task_name translation


python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-80" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name translation
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-20" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name translation
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-5" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name translation
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-1" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name translation

###
# cls - value
###a
python translationese_eval.py --model_path "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name translation
python translationese_eval.py --model_path "sentence-transformers/roberta-base-nli-stsb-mean-tokens" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name translation
python translationese_eval.py --model_path "sentence-transformers/stsb-bert-base" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name translation
python translationese_eval.py --model_path "sentence-transformers/average_word_embeddings_glove.6B.300d" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name translation

python translationese_eval.py --model_path "sentence-transformers/all-mpnet-base-v2" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --task_name translation
python translationese_eval.py --model_path "sentence-transformers/LaBSE" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --task_name translation
python translationese_eval.py --model_path "nomic-ai/nomic-embed-text-v1" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --task_name translation

python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-80" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name translation
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-20" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name translation
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-5" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name translation
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-1" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name translation

#######
###
# Paraphrase
###
#######

###
# knn - value
###
python translationese_eval.py --model_path "sentence-transformers/average_word_embeddings_glove.6B.300d" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name paraphrase
python translationese_eval.py --model_path "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name paraphrase
python translationese_eval.py --model_path "sentence-transformers/roberta-base-nli-stsb-mean-tokens" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name paraphrase
python translationese_eval.py --model_path "sentence-transformers/stsb-bert-base" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name paraphrase

python translationese_eval.py --model_path "sentence-transformers/all-mpnet-base-v2" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --task_name paraphrase
python translationese_eval.py --model_path "sentence-transformers/LaBSE" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --task_name paraphrase
python translationese_eval.py --model_path "nomic-ai/nomic-embed-text-v1" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --task_name paraphrase

python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-80" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name paraphrase
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-20" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name paraphrase
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-5" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name paraphrase
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-1" --training_type knn --ouput_dir ./knn_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name paraphrase


###
# cls - value
###a

###
# cls - value
###a
python translationese_eval.py --model_path "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name paraphrase
python translationese_eval.py --model_path "sentence-transformers/roberta-base-nli-stsb-mean-tokens" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name paraphrase
python translationese_eval.py --model_path "sentence-transformers/stsb-bert-base" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name paraphrase
python translationese_eval.py --model_path "sentence-transformers/average_word_embeddings_glove.6B.300d" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 20 --num_qas 1 --task_name paraphrase

python translationese_eval.py --model_path "sentence-transformers/all-mpnet-base-v2" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --task_name paraphrase
python translationese_eval.py --model_path "sentence-transformers/LaBSE" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --task_name paraphrase
python translationese_eval.py --model_path "nomic-ai/nomic-embed-text-v1" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --task_name paraphrase

python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-80" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name paraphrase
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-20" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name paraphrase
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-5" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name paraphrase
python translationese_eval.py --model_path "CAiRE/UniVaR-lambda-1" --training_type cls --ouput_dir ./cls_europarl --cache_dir ./cache_europarl --batch_size 512 --n_epochs 20 --learning_rate 2e-3 --n_neighbors 50 --num_qas 1 --task_name paraphrase