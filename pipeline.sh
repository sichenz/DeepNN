# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

# Pipeline

## Input
CONFIG=configs/config.yaml
LABEL=bootstrap
FOLDER=paper
cd /Users/princess/Documents/RA/DeepNN #change this

## Data preparation
python -m data.simulated-data -d config-05  -c $CONFIG -l $LABEL
python -m data.build-prediction-master      -c $CONFIG -l $LABEL
python -m data.word2vec                     -c $CONFIG -l $LABEL
python -m uplift.true                       -c $CONFIG -l $LABEL

## Model baselines
python -m baselines.frequency               -c $CONFIG -l $LABEL
python -m baselines.logit-cross-by-j        -c $CONFIG -l $LABEL
python -m baselines.lightgbm-cross-by-j     -c $CONFIG -l $LABEL
python -m baselines.lightgbm-cat-cross-by-j -c $CONFIG -l $LABEL
python -m baselines.run-mxl                 -c $CONFIG -l $LABEL

## DNN
python -m dnn.train                         -c $CONFIG -l $LABEL -p
python -m dnn.predict                       -c $CONFIG -l $LABEL -t dnn/config_model.yaml -e 99

## Probability simulations
python -m uplift.logit-cross-by-j           -c $CONFIG -l $LABEL
python -m uplift.lightgbm-cross-by-j        -c $CONFIG -l $LABEL
python -m uplift.lightgbm-cat-cross-by-j    -c $CONFIG -l $LABEL
python -m uplift.dnn                        -c $CONFIG -l $LABEL -m model_010
python -m uplift.combine-prob-estimates     -c $CONFIG -l $LABEL

## Evaluations
python -m results.run-benchmark             -c $CONFIG -l $LABEL
python -m uplift.elasticities               -c $CONFIG -l $LABEL
python -m uplift.coupon-optim               -c $CONFIG -l $LABEL

## Figure 2
python -m paper.figure-2-data               -c $CONFIG
python -m paper.figure-2-mxl                -c $CONFIG

## Paper results
python -m paper.table_1                     -c $CONFIG -f $FOLDER -m log-loss
python -m paper.table_1                     -c $CONFIG -f $FOLDER -m kl-divergence
python -m paper.table_2                     -c $CONFIG -f $FOLDER
python -m paper.table_3                     -c $CONFIG -f $FOLDER
python -m paper.table_4                     -c $CONFIG -f $FOLDER
python -m paper.figure-2-plot               -c $CONFIG
python -m paper.figure_3                    -c $CONFIG
