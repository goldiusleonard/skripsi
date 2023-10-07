TRAIN="
embedding_data/divt_mobilevits_OMItoC/casia_mfsd.pkl \
"

TEST="
embedding_data/divt_mobilevits_OMItoC/casia_mfsd.pkl \
"

# ------------------------------------

SAVE_FOLDER=F:/skripsi/FAS-Skripsi-4/eval_result/divt_mobilevits_OMItoC/
mkdir $SAVE_FOLDER

python F:/skripsi/FAS-Skripsi-4/evaluator/metric/eval_knn_agi_remove_feature.py \
    --train $TRAIN \
    --test $TEST \
    --n_clusters 50 \
    --save_folder $SAVE_FOLDER \
    --test_main_th 0.3156813085079193 \
