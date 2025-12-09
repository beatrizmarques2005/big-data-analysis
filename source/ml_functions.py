from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
import pyspark.sql.functions as F


# ------------------------------
# 1. ROC-AUC
# ------------------------------
def get_auc(preds, label_col="label", raw_pred_col="rawPrediction"):
    evaluator = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol=raw_pred_col,
        metricName="areaUnderROC"
    )
    return evaluator.evaluate(preds)


# ------------------------------
# 2. F1 Score
# ------------------------------
def get_f1(preds, label_col="label", pred_col="prediction"):
    evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=pred_col,
        metricName="f1"
    )
    return evaluator.evaluate(preds)


# ------------------------------
# 3. Confusion Matrix
# ------------------------------
def get_confusion_matrix(preds, label_col="label", pred_col="prediction"):
    cm = preds.groupBy(label_col, pred_col).count().orderBy(label_col, pred_col)
    return cm