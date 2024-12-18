# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

suppressMessages({
  require(data.table)
  library(arrow)
  library(argparse)
  require(magrittr)
  require(testthat)
  require(stargazer)
  require(yaml)
  require(glue)
})

ll = function (p, y, eps=1e-12) {
  peps = pmax(pmin(p, 1-eps), eps)  # clip probabilities
  -(y*log(peps) + (1-y)*log(1-peps))
}


# config
parser = ArgumentParser()
parser$add_argument("-c", "--c", type="character", help="config file")
args = parser$parse_args()

config = read_yaml(args$c)
path_data_raw = path.expand(config[["path_data_raw"]])
path_data = path.expand(config[["path_data"]])
path_results = path.expand(config[["path_results"]])


# load data

# model predictions for hold-out weeks
file_predictions =
  "{path_results}/predictions_overview.parquet" %>%
  glue %T>%
  print

predictions =
  read_parquet(file_predictions) %>%
  data.table %>%
  .[,
    list(
        i=as.integer(i),
        j=as.integer(j),
        t=as.integer(t),
        y=as.integer(y),
        it_has_coupon=as.integer(it_has_coupon),
        dnn=as.numeric(dnn),
        mxl=as.numeric(mxl),
        lightgbm=as.numeric(lightgbm),
        logit=as.numeric(logit)
    )
  ]

# map between product IDs and category IDs
# (according to retailer's category definition)
data_j =
  "{path_data_raw}/data_j.csv" %>%
  glue %>%
  fread %>%
  .[, list(j, category)]

# map between product IDs and brand IDs (the new id for the model)
# (according to retailer's brand definition)
product_id_map =
  "{path_data}/product_id_map.parquet" %>%
  glue %>%
  read_parquet(file=.) %>%
  data.table

# map between brand IDs (the level and category IDs
product_id_map_category = merge(product_id_map, data_j, by="j", all.x=TRUE)
expect_false(product_id_map_category[, any(is.na(category))])
# only aggregate products within categories
expect_true(product_id_map_category[, uniqueN(category), by=j_2][, all(V1==1)])
j_category = unique(product_id_map_category[, list(j=j_2, category)])

# loyalty card data
baskets =
  "{path_data}/baskets.parquet" %>%
  glue %>%
  read_parquet(file=.) %>%
  data.table


# compute ipt (as feature in loss analysis)
expect_true(baskets[, all(j %in% j_category[, j])])
baskets = merge(baskets, j_category, by="j")
ipt_ic = baskets[, list(ipt=(max(t)+1)/.N), by=list(i,category)]


# build regression base table
base_table =
  predictions %>%
	copy %>%
  merge(data_j, by="j")


# build last purchase indicator
# (requires that data is sorted)
setorder(base_table, i, j, t)
base_table[, y_l1 := shift(y, 1), by=c("i", "j")]


# build coupon indicator
base_table[, coupon := as.integer(it_has_coupon)]


# build last purchase indicator for category
base_table[, y_l1_c := sum(y_l1), by=list(i, t, category)]


# remove first week because of shift
base_table = base_table[!is.na(y_l1)]


# compute loss
base_table[, ll_dnn := ll(dnn, y)]
base_table[, ll_mxl := ll(mxl, y)]
base_table[, ll_gbm := ll(lightgbm, y)]
base_table[, ll_bl := ll(logit, y)]
expect_false(any(is.na(base_table)))


# add more explanatory variables
base_table[, dc := sprintf("dc%04d", .GRP), by=category]
base_table[, int := 1]
expect_true(all(base_table[, unique(i)] %in% ipt_ic[, unique(as.integer(i))]))
base_table = merge(base_table, ipt_ic, by=c("i", "category"))


# regression analysis: DNN loss vs. baselines models

# 1) MXL
dtm_mxl = rbindlist(list(
  base_table[, list(int=1, dnn=1, ll=100*ll_dnn, coupon, y_l1_c, ipt, dc)],
  base_table[, list(int=1, dnn=0, ll=100*ll_mxl, coupon, y_l1_c, ipt, dc)]
))
res_mxl_1 = lm(data=dtm_mxl, ll ~ int + dnn -1)
res_mxl_2 = lm(data=dtm_mxl, ll ~ int + dnn + coupon + y_l1_c + dnn:ipt + dnn:coupon + dnn:y_l1_c + ipt + dc -1)

# 2) GBM
dtm_gbm = rbindlist(list(
  base_table[, list(int=1, dnn=1, ll=100*ll_dnn, coupon, y_l1_c, ipt, dc)],
  base_table[, list(int=1, dnn=0, ll=100*ll_gbm, coupon, y_l1_c, ipt, dc)]
))
res_gbm_1 = lm(data=dtm_gbm, ll ~ int + dnn -1)
res_gbm_2 = lm(data=dtm_gbm, ll ~ int + dnn + coupon + y_l1_c + dnn:ipt + dnn:coupon + dnn:y_l1_c + ipt + dc -1)

# 3) BINARY LOGIT
dtm_bl = rbindlist(list(
  base_table[, list(int=1, dnn=1, ll=100*ll_dnn, coupon, y_l1_c, ipt, dc)],
  base_table[, list(int=1, dnn=0, ll=100*ll_bl, coupon, y_l1_c, ipt, dc)]
))
res_bl_1 = lm(data=dtm_bl, ll ~ int + dnn -1)
res_bl_2 = lm(data=dtm_bl, ll ~ int + dnn + coupon + y_l1_c + dnn:ipt + dnn:coupon + dnn:y_l1_c + ipt + dc -1)


# TABLE 7
stargazer(
    res_mxl_1,
    res_mxl_2,
    res_gbm_1,
    res_gbm_2,
    res_bl_1,
    res_bl_2,
    type= "text",
    omit="dcdc",
    omit.stat="all"
)

dir.create(glue("{path_results}/paper"), showWarnings=FALSE)
sink(glue("{path_results}/paper/table_7.html"))
  stargazer(
      res_mxl_1,
      res_mxl_2,
      res_gbm_1,
      res_gbm_2,
      res_bl_1,
      res_bl_2,
      type="html",
      omit="dcdc",
      omit.stat="all"
  )
sink()

