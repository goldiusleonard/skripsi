from argparse import ArgumentParser
from cProfile import label
import pathlib
from pathlib import Path
import pickle
import dataclasses
import random
from typing import Dict, List, Union
import numpy as np
from fas_eval.evaluators.binary import BinarySpoofEvaluator
import cv2
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from pytorch_metric_learning.losses.centroid_triplet_loss import CentroidTripletLoss


@dataclasses.dataclass
class InferenceData:
    main_embs: np.ndarray
    labels: Union[List[int], np.ndarray]
    img_paths: List[str]
    main_scores: np.ndarray
    _is_empty: bool = False

    @classmethod
    def create_empty(cls):
        return cls(
            main_embs=np.empty(0),
            labels=[],
            img_paths=[],
            main_scores=np.empty(0),
            _is_empty=True,
        )

    def _handle_add_empty(self, other):
        if self._is_empty:
            self.main_embs = np.empty((0, *other.main_embs.shape[1:]))
            self.main_scores = np.empty((0, *other.main_scores.shape[1:]))
            print("is empty is called")
        else:
            return

        self._is_empty = False

    def __add__(self, other):
        self._handle_add_empty(other)

        main_embs = np.concatenate((self.main_embs, other.main_embs), axis=0)
        labels = np.concatenate((self.labels, other.labels), axis=0)
        img_paths = self.img_paths + other.img_paths
        main_scores = np.concatenate((self.main_scores, other.main_scores), axis=0)

        return InferenceData(
            main_embs,
            labels,
            img_paths,
            main_scores,
        )


def save_eer_curve(evaluator: BinarySpoofEvaluator, save_path: str, **kwargs):
    eer_curve = evaluator.metrics["EER_CURVE"]

    eer_df = pd.DataFrame(eer_curve)
    eer_df.to_csv(save_path, index=False, **kwargs)


def save_custom_eval_csv(
    savepath,
    raw_prediction,
    thresholded_prediction,
    ground_truth,
    prediction_status,
    **column_datas,
):
    data = {
        "raw_prediction": raw_prediction,
        "thresholded_prediction": thresholded_prediction,
        "ground_truth": ground_truth,
        "prediction_status": prediction_status,
    }

    data.update(column_datas)

    pd.DataFrame(data).to_csv(savepath, index_label="index")


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--train", type=str, nargs="+", metavar="train_data_pickle_path", required=True
    )
    parser.add_argument(
        "--test", type=str, nargs="+", metavar="test_data_pickle_path", required=True
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        required=True,
    )
    parser.add_argument("--save_folder", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--test_main_th", type=float, default=-1.0)
    args = parser.parse_args()

    return args


def read_data(picklepath):
    with open(picklepath, "rb") as pklfile:
        data = pickle.load(pklfile)

    main_embs = data["main_embs"]
    labels = data["labels"]
    img_paths = data["img_paths"]
    main_scores = data["main_scores"]

    return InferenceData(
        main_embs,
        labels,
        img_paths,
        main_scores,
    )


def repeat_to_list(el, length: int):
    return [el for _ in range(length)]


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_datas: Dict[str, InferenceData] = dict()
    for pickle_path in args.train:
        train_data = read_data(pickle_path)
        train_datas[pickle_path] = train_data

    with open(args.train[0], "rb") as pickle_file:
        train_data = pickle.load(pickle_file)

    test_datas: Dict[str, InferenceData] = dict()
    for pickle_path in args.test:
        test_data = read_data(pickle_path)
        test_datas[pickle_path] = test_data

    # ---------------------------------------------------------------------------- #
    #                             train data evaluation                            #
    # ---------------------------------------------------------------------------- #
    train_data_combined = InferenceData.create_empty()
    pickle_paths_col = []

    for pickle_path, train_data in train_datas.items():
        train_data_combined = train_data_combined + train_data
        pickle_paths_col.append(repeat_to_list(pickle_path, len(train_data.labels)))

    # --------------------------- main score evaluation -------------------------- #
    main_predictions_train = train_data_combined.main_scores
    ground_truths_train = train_data_combined.labels

    # handle 1d-sigmoid, 2d-sigmoid or softmax output
    if len(main_predictions_train.shape) == 2:
        if main_predictions_train.shape[1] == 1:
            main_predictions_train = np.reshape(main_predictions_train, -1)
        else:
            # use the second output as live score
            main_predictions_train = main_predictions_train[:, 1]

    if main_predictions_train.shape[0] != len(ground_truths_train):
        raise RuntimeError(
            f"Mismatch shape of main_prediction_train ({main_predictions_train.shape}), expected {len(ground_truths_train)}"
        )

    main_evaluator_train = BinarySpoofEvaluator(
        raw_predictions=main_predictions_train,
        ground_truths=ground_truths_train,
        interp="nearest",
    )
    main_evaluator_train.calculate_metrics(threshold_from="eer")
    main_evaluator_train.save_csv(
        path=args.save_folder / "main_evaluator_train.csv",
        paths=train_data_combined.img_paths,
    )
    main_evaluator_train.save_json(
        path=args.save_folder / "main_evaluator_train.json",
    )
    save_eer_curve(
        main_evaluator_train, args.save_folder / "main_evaluator_train-eer.csv"
    )

    # -------------------------- border score evaluation ------------------------- #
    # border_predictions_train = train_data.border_scores
    # ground_truths_train = train_data.labels

    # handle 1d-sigmoid, 2d-sigmoid or softmax output
    # if len(border_predictions_train.shape) == 2:
    #     if border_predictions_train.shape[1] == 1:
    #         border_predictions_train = np.reshape(border_predictions_train, -1)
    #     else:
    #         # use the second output as live score
    #         border_predictions_train = border_predictions_train[:, 1]

    # assert border_predictions_train.shape[0] == len(ground_truths_train)

    # border_evaluator_train = BinarySpoofEvaluator(
    #     raw_predictions=border_predictions_train,
    #     ground_truths=ground_truths_train,
    #     interp="nearest",
    # )
    # border_evaluator_train.calculate_metrics(threshold_from="eer")
    # border_evaluator_train.save_csv(
    #     path=args.save_folder / "border_evaluator_train.csv",
    #     paths=train_data.img_paths,
    # )
    # border_evaluator_train.save_json(
    #     path=args.save_folder / "border_evaluator_train.json",
    # )
    # save_eer_curve(border_evaluator_train, args.save_folder / "border_evaluator_train-eer.csv")

    # ------------------- main-filtered border score evaluation ------------------ #
    # border_preds_filtered_by_main_train = main_predictions_train
    # idx_classified_as_spoof_by_main_train = (
    #     main_evaluator_train.thresholded_predictions == 0
    # )
    # border_preds_filtered_by_main_train[idx_classified_as_spoof_by_main_train] = 0

    # filtered_border_evaluator_train = BinarySpoofEvaluator(
    #     raw_predictions=border_preds_filtered_by_main_train,
    #     ground_truths=ground_truths_train,
    #     interp="nearest",
    # )
    # filtered_border_evaluator_train.calculate_metrics(threshold_from="eer")
    # filtered_border_evaluator_train.save_csv(
    #     path=args.save_folder / "filtered_border_evaluator_train.csv",
    #     paths=train_data.img_paths,
    # )
    # filtered_border_evaluator_train.save_json(
    #     path=args.save_folder / "filtered_border_evaluator_train.json",
    # )
    # save_eer_curve(filtered_border_evaluator_train, args.save_folder / "filtered_border_evaluator_train-eer.csv")

    # ------------------------ train cluster construction ------------------------ #
    train_emb = train_data_combined.main_embs.astype(np.float32)

    train_live_emb = train_emb[ground_truths_train == 1]
    train_spoof_emb = train_emb[ground_truths_train == 0]

    train_clusters = []
    train_cluster_labels = []

    kmeans_real = KMeans(
        n_clusters=args.n_clusters, init="random", n_init=100, random_state=args.seed
    )
    kmeans_real.fit(train_live_emb)
    for cent in kmeans_real.cluster_centers_:
        train_clusters.append(cent)
        train_cluster_labels.append(1)

    kmeans_spoof = KMeans(
        n_clusters=args.n_clusters, init="random", n_init=100, random_state=args.seed
    )
    kmeans_spoof.fit(train_spoof_emb)
    for cent in kmeans_spoof.cluster_centers_:
        train_clusters.append(cent)
        train_cluster_labels.append(0)

    train_clusters = np.stack(train_clusters).astype(np.float32)
    train_cluster_labels = np.stack(train_cluster_labels)

    train_clust_data = np.concatenate(
        (train_clusters, np.asarray(train_cluster_labels).reshape(-1, 1)),
        axis=1,
    ).astype(np.float32)
    train_df = pd.DataFrame(train_clust_data)
    train_df.to_csv(str(args.save_folder / "clust_train_data.csv"), index=False)

    train_clust_dict = {
        "embs_train": train_clusters,
        "labels_train": train_cluster_labels,
    }
    with open(args.save_folder / "train_clusters_data.pkl", "wb") as pklfile:
        pickle.dump(train_clust_dict, pklfile)

    train_data_dict = dataclasses.asdict(train_data_combined)
    train_data_dict = train_data
    with open(args.save_folder / "train_data.pkl", "wb") as pklfile:
        pickle.dump(train_data_dict, pklfile)

    # ---------------------------------------------------------------------------- #
    #                              test data evaluation                            #
    # ---------------------------------------------------------------------------- #
    test_data_combined = InferenceData.create_empty()
    pickle_paths_col = []

    for pickle_path, test_data in test_datas.items():
        test_data_combined = test_data_combined + test_data
        pickle_paths_col.append(repeat_to_list(pickle_path, len(test_data.labels)))

    # --------------------------- main score evaluation -------------------------- #
    main_predictions_test = test_data_combined.main_scores
    ground_truths_test = test_data_combined.labels

    # handle 1d-sigmoid, 2d-sigmoid or softmax output
    if len(main_predictions_test.shape) == 2:
        if main_predictions_test.shape[1] == 1:
            main_predictions_test = np.reshape(main_predictions_test, -1)
        else:
            # use the second output as live score
            main_predictions_test = main_predictions_test[:, 1]

    assert main_predictions_test.shape[0] == len(ground_truths_test)

    main_evaluator_test = BinarySpoofEvaluator(
        raw_predictions=main_predictions_test,
        ground_truths=ground_truths_test,
        interp="nearest",
    )

    if args.test_main_th == -1:
        threshold_from = "eer"
    else:
        threshold_from = args.test_main_th
    print("main_evaluator_test", "threshold from", threshold_from)

    main_evaluator_test.calculate_metrics(threshold_from=threshold_from)
    main_evaluator_test.save_csv(
        path=args.save_folder / "main_evaluator_test.csv",
        paths=test_data_combined.img_paths,
    )
    main_evaluator_test.save_json(
        path=args.save_folder / "main_evaluator_test.json",
    )
    save_eer_curve(
        main_evaluator_test, args.save_folder / "main_evaluator_test-eer.csv"
    )

    # -------------------------- border score evaluation ------------------------- #
    # border_predictions_test = test_data_combined.border_scores
    # ground_truths_test = test_data_combined.labels

    # # handle 1d-sigmoid, 2d-sigmoid or softmax output
    # if len(border_predictions_test.shape) == 2:
    #     if border_predictions_test.shape[1] == 1:
    #         border_predictions_test = np.reshape(border_predictions_test, -1)
    #     else:
    #         # use the second output as live score
    #         border_predictions_test = border_predictions_test[:, 1]

    # assert border_predictions_test.shape[0] == len(ground_truths_test)

    # border_evaluator_test = BinarySpoofEvaluator(
    #     raw_predictions=border_predictions_test,
    #     ground_truths=ground_truths_test,
    #     interp="nearest",
    # )

    # if args.test_fborder_th == -1:
    #     threshold_from = "eer"
    # else:
    #     threshold_from = args.test_fborder_th
    # print("border_evaluator_test", "threshold from", threshold_from)

    # border_evaluator_test.calculate_metrics(threshold_from=threshold_from)
    # border_evaluator_test.save_csv(
    #     path=args.save_folder / "border_evaluator_test.csv",
    #     paths=test_data_combined.img_paths,
    # )
    # border_evaluator_test.save_json(
    #     path=args.save_folder / "border_evaluator_test.json",
    # )
    # save_eer_curve(border_evaluator_test, args.save_folder / "border_evaluator_test-eer.csv")

    # ------------------- main-filtered border score evaluation ------------------ #
    # border_preds_filtered_by_main_test = border_predictions_test
    # idx_classified_as_spoof_by_main_test = (
    #     main_evaluator_test.thresholded_predictions == 0
    # )
    # border_preds_filtered_by_main_test[idx_classified_as_spoof_by_main_test] = 0

    # filtered_border_evaluator_test = BinarySpoofEvaluator(
    #     raw_predictions=border_preds_filtered_by_main_test,
    #     ground_truths=ground_truths_test,
    #     interp="nearest",
    # )

    # if args.test_fborder_th == -1:
    #     threshold_from = "eer"
    # else:
    #     threshold_from = args.test_fborder_th
    # print("filtered_border_evaluator_test", "threshold from", threshold_from)

    # filtered_border_evaluator_test.calculate_metrics(threshold_from=threshold_from)
    # filtered_border_evaluator_test.save_csv(
    #     path=args.save_folder / "filtered_border_evaluator_test.csv",
    #     paths=test_data_combined.img_paths,
    # )
    # filtered_border_evaluator_test.save_json(
    #     path=args.save_folder / "filtered_border_evaluator_test.json",
    # )
    # save_eer_curve(filtered_border_evaluator_test, args.save_folder / "filtered_border_evaluator_test-eer.csv")

    # ---------------------------- test knn evaluation --------------------------- #
    knn = cv2.ml.KNearest_create()
    # print(train_clusters.shape, train_clusters.dtype)
    # print(train_cluster_labels.shape, train_cluster_labels.dtype)
    train_cluster_labels = train_cluster_labels.reshape(-1, 1)
    knn.train(train_clusters, cv2.ml.ROW_SAMPLE, train_cluster_labels)

    npcers = []
    apcers = []

    n_neigh = [1, 3, 5, 7, 9, 11]
    for k in n_neigh:
        _, results, _, _ = knn.findNearest(
            test_data_combined.main_embs.astype(np.float32), k
        )

        results = np.reshape(results, -1)
        assert len(results) == len(ground_truths_test)

        # ----------------------------- regular knn test ----------------------------- #
        knn_evaluator_test = BinarySpoofEvaluator(
            raw_predictions=results,
            ground_truths=ground_truths_test,
            interp="nearest",
        )
        knn_evaluator_test.calculate_metrics(threshold_from="none")
        knn_evaluator_test.save_csv(
            path=args.save_folder / f"knn_evaluator_test_k{k}.csv",
            paths=test_data_combined.img_paths,
        )
        knn_evaluator_test.save_json(
            path=args.save_folder / f"knn_evaluator_test_k{k}.json",
        )
        npcers.append(knn_evaluator_test.metrics["NPCER"])
        apcers.append(knn_evaluator_test.metrics["APCER"])

        # ------------------------ pipeline filtered knn test ------------------------ #
        # results_filtered_by_border = results
        # idx_classifier_as_spoof_by_pipeline = (
        #     filtered_border_evaluator_test.thresholded_predictions == 0
        # )
        # results_filtered_by_border[idx_classifier_as_spoof_by_pipeline] = 0
        # knn_evaluator_filtered_test = BinarySpoofEvaluator(
        #     raw_predictions=results, ground_truths=ground_truths_test, interp="nearest",
        # )
        # knn_evaluator_filtered_test.calculate_metrics(threshold_from="none")
        # knn_evaluator_filtered_test.save_csv(
        #     path=args.save_folder / f"knn_evaluator_filtered_test{k}.csv",
        #     paths=test_data_combined.img_paths,
        # )
        # knn_evaluator_filtered_test.save_json(
        #     path=args.save_folder / f"knn_evaluator_filtered_test{k}.json",
        # )
        # npcers_filtered.append(knn_evaluator_filtered_test.metrics["NPCER"])
        # apcers_filtered.append(knn_evaluator_filtered_test.metrics["APCER"])

        # save_custom_eval_csv(
        #     savepath=args.save_folder / f"knn_evaluator_filtered_test{k}_full.csv",
        #     raw_prediction=knn_evaluator_test.raw_predictions,
        #     thresholded_prediction=knn_evaluator_test.thresholded_predictions,
        #     ground_truth=knn_evaluator_test.ground_truths,
        #     prediction_status=knn_evaluator_test.prediction_status,
        #     main_score=main_evaluator_test.raw_predictions,
        #     main_score_pred_stat=main_evaluator_test.prediction_status,
        #     knn_prediction=knn_evaluator_test.raw_predictions,
        #     knn_pred_stat=knn_evaluator_test.prediction_status,
        #     path=test_data_combined.img_paths,
        # )

    fig, ax = plt.subplots()
    ax: Axes

    viz_df = pd.DataFrame(
        {
            "k": n_neigh,
            "npcer": npcers,
            "apcer": apcers,
        }
    )
    viz_df_melt = pd.melt(viz_df, id_vars="k")
    sns.barplot(x="k", y="value", hue="variable", data=viz_df_melt, ax=ax)

    fig.savefig(str(args.save_folder / "metric.png"))

    test_data_dict = dataclasses.asdict(test_data_combined)
    with open(args.save_folder / "test_data.pkl", "wb") as pklfile:
        print(test_data_dict.keys())
        pickle.dump(test_data_dict, pklfile)


if __name__ == "__main__":
    main()
