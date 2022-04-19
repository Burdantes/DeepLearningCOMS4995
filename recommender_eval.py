# set the environment path to find Recommenders

import sys

import pyspark
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, IntegerType, LongType, StructType, StructField
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import HashingTF, CountVectorizer, VectorAssembler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.spark_splitters import spark_random_split
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkDiversityEvaluation
from recommenders.utils.spark_utils import start_or_get_spark

from pyspark.sql.window import Window
import pyspark.sql.functions as F

import numpy as np
import pandas as pd
from typing import Optional

print("System version: {}".format(sys.version))
print("Spark version: {}".format(pyspark.__version__))

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# user, item column names
COL_USER="UserId"
COL_ITEM="MovieId"
COL_RATING="Rating"
COL_TITLE="Title"
COL_GENRE="Genre"

# the following settings work well for debugging locally on VM - change when running on a cluster
# set up a giant single executor with many threads and specify memory cap

spark = start_or_get_spark("ALS PySpark", memory="16g")
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")
spark.conf.set("spark.sql.crossJoin.enabled", "true")

# Note: The DataFrame-based API for ALS currently only supports integers for user and item ids.
schema = StructType(
    (
        StructField(COL_USER, IntegerType()),
        StructField(COL_ITEM, IntegerType()),
        StructField(COL_RATING, FloatType()),
        StructField("Timestamp", LongType()),
    )
)

data = movielens.load_spark_df(spark, size=MOVIELENS_DATA_SIZE, schema=schema, title_col=COL_TITLE, genres_col=COL_GENRE)
data.show()

train_df, test_df = spark_random_split(data.select(COL_USER, COL_ITEM, COL_RATING), ratio=0.75, seed=123)
print ("N train_df", train_df.cache().count())
print ("N test_df", test_df.cache().count())

users = train_df.select(COL_USER).distinct()
items = train_df.select(COL_ITEM).distinct()
user_item = users.crossJoin(items)

header = {
    "userCol": COL_USER,
    "itemCol": COL_ITEM,
    "ratingCol": COL_RATING,
}


als = ALS(
    rank=10,
    maxIter=15,
    implicitPrefs=False,
    regParam=0.05,
    coldStartStrategy='drop',
    nonnegative=False,
    seed=42,
    **header
)

with Timer() as train_time:
    model = als.fit(train_df)

print("Took {} seconds for training.".format(train_time.interval))

# Score all user-item pairs
dfs_pred = model.transform(user_item)

# Remove seen items.
dfs_pred_exclude_train = dfs_pred.alias("pred").join(
	train_df.alias("train"),
	(dfs_pred[COL_USER] == train_df[COL_USER]) & (dfs_pred[COL_ITEM] == train_df[COL_ITEM]),
	how='outer'
)

top_all = dfs_pred_exclude_train.filter(dfs_pred_exclude_train["train.Rating"].isNull()) \
	.select('pred.' + COL_USER, 'pred.' + COL_ITEM, 'pred.' + "prediction")

print(top_all.count())

window = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())
top_k_reco = top_all.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= TOP_K).drop("rank")

print(top_k_reco.count())

# random recommender
window = Window.partitionBy(COL_USER).orderBy(F.rand())

# randomly generated recommendations for each user
pred_df = (
  train_df
  # join training data with all possible user-item pairs (seen in training)
  .join(user_item,
        on=[COL_USER, COL_ITEM],
        how="right"
  )
  # get user-item pairs that were not seen in the training data
  .filter(F.col(COL_RATING).isNull())
  # count items for each user (randomly sorting them)
  .withColumn("score", F.row_number().over(window))
  # get the top k items per user
  .filter(F.col("score") <= TOP_K)
  .drop(COL_RATING)
)


def get_ranking_results(ranking_eval):
	metrics = {
		"Precision@k": ranking_eval.precision_at_k(),
		"Recall@k": ranking_eval.recall_at_k(),
		"NDCG@k": ranking_eval.ndcg_at_k(),
		"Mean average precision": ranking_eval.map_at_k()

	}
	return metrics


def get_diversity_results(diversity_eval):
	metrics = {
		"catalog_coverage": diversity_eval.catalog_coverage(),
		"distributional_coverage": diversity_eval.distributional_coverage(),
		"novelty": diversity_eval.novelty(),
		"diversity": diversity_eval.diversity(),
		"serendipity": diversity_eval.serendipity()
	}
	return metrics


def generate_summary(data, algo, k, ranking_metrics, diversity_metrics):
	summary = {"Data": data, "Algo": algo, "K": k}

	if ranking_metrics is None:
		ranking_metrics = {
			"Precision@k": np.nan,
			"Recall@k": np.nan,
			"nDCG@k": np.nan,
			"MAP": np.nan,
		}
	summary.update(ranking_metrics)
	summary.update(diversity_metrics)
	return summary