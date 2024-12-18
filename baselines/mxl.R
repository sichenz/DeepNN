# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

# libraries
suppressMessages({
  library(magrittr)
  library(data.table)
  library(foreach)
  library(doParallel)
  library(bayesm)
  library(glue)
  library(reshape2)
  library(pracma)
  library(argparse)
  library(yaml)
  library(ggplot2)
  library(testthat)
  library(reticulate)
})

# python integration
path_virtualenv = Sys.getenv("PATH_VIRTUALENV")
sprintf("path_virtualenv = %s", path_virtualenv) %>% cat(fill=TRUE)
use_virtualenv(path_virtualenv, required=TRUE)
pd = import("pandas", "pd")
pq = import("pyarrow.parquet", "pq")

# script arguments
parser = ArgumentParser()
parser$add_argument("-c", "--c", default='configs/config.yaml', type="character", help="config")
parser$add_argument("-s", "--s", type="integer", help="data set id", metavar="number")
parser$add_argument("-u", "--u", action='store_false', help="do uplift")
parser$add_argument("-p", "--p", type="character", help="data path")
parser$add_argument("-y", "--y", type="integer", help="modulo divisor")
parser$add_argument("-z", "--z", type="integer", help="modulo remainder")
args = parser$parse_args()

if (!is.null(args$y) & !is.null(args$z)) {
    if ((args$s %% args$y) != args$z) {
        quit('no')
    }
}

cat(sprintf('config = %s', args$c), fill=TRUE)

if (is.null(args[['s']])) {
  suffix = ''
} else {
  suffix = sprintf("_%03d", args[['s']])
}
cat(sprintf('suffix = %s', suffix), fill=TRUE)


## GLOBAL DEFINITIONS
I = 2000
config = read_yaml(args$c)
n_cores = config[['n_cores']][['mxl']]
sprintf('n_cores = %d', n_cores) %>% cat(fill=TRUE)
train_params = config[['training']]
t_train_start = train_params[['time_first']]
t_train_end = train_params[['time_last']]
cat(glue("train weeks = [{t_train_start}, ..., {t_train_end}]"), fill=TRUE)
t_test_start = train_params[['time_last']]+1
t_test_end = train_params[['time_last']]+10
cat(glue("test weeks = [{t_test_start}, ..., {t_test_end}]"), fill=TRUE)
t_prob_uplift = t_test_end + 1
cat(glue("uplift week = {t_prob_uplift}"), fill=TRUE)
T_train = t_train_end-t_train_start+1
T_test = t_test_end-t_test_start+1
result_folder = 'mxl-inventory'

avg_windows = config[['training']][['avg_windows']]

if (!is.null(args$p)) {
    cat(glue("set path to {args$p}"), fill=TRUE)
    config['path_data'] = args$p
}
set.seed(config$global_seed)

I_prob_uplift = as.integer(gsub('_', '', config[['coupons']][['I']]))
d_prob_uplift = config[['coupons']][['discount']]

R = config[['mxl']][['R']]
keep = config[['mxl']][['keep']]


## MAIN -- LOOP OVER DATA SETS

# output dir
path_data =
    config[['path_data']] %>%
    sub(pattern="\\$HOME", replacement=Sys.getenv("HOME"), x=.) %>%
    paste0(suffix)
sprintf("path_data = %s", path_data) %>% cat(fill=TRUE)

file_uplift_true =
    "{path_data}/prob_uplift/total_prob_true.parquet" %>%
    glue
file_uplift_true %>%
    file.exists %>%
    expect_true

dir.create(file.path(path_data, 'baselines'), showWarnings=FALSE)

file_result = file.path(path_data, "baselines/MXL_inventory.parquet")
file_prob_uplift = file.path(path_data, "prob_uplift", 'total_prob_mxl_inv.parquet')

if (file.exists(file_result)) {
    cat(glue("result already exists, skipping {path_data}"), fill=TRUE)
    quit('no')
} else {
    system(glue("touch {file_result}"))
}

# load data
tmp_baskets = pd$read_parquet(file.path(path_data, "baskets.parquet"))
baskets = data.table(
  i=as.integer(tmp_baskets$i$values),
  j=as.integer(tmp_baskets$j$values),
  t=as.integer(tmp_baskets$t$values),
  p_jc=as.numeric(tmp_baskets$p_jc$values),
  price_paid=as.numeric(tmp_baskets$price_paid$values),
  d_ijt=as.numeric(tmp_baskets$d_ijt$values)
)
baskets = baskets[i<I]

file_actions = file.path(path_data, "action.parquet")
if (file.exists(file_actions)) {
    tmp_actions = pd$read_parquet(file.path(path_data, "action.parquet"))
    actions = data.table(
      i=as.integer(tmp_actions$i$values),
      j=as.integer(tmp_actions$j$values),
      t=as.integer(tmp_actions$t$values),
      nc=as.integer(tmp_actions$nc$values),
      discount=as.numeric(tmp_actions$discount$values)
    )
    actions = actions[i<I]
    use_discounts = TRUE
} else {
    use_discounts = FALSE
    cat('warning -- not using discounts', fill=TRUE)
}
brand_category =
    file.path(path_data, 'data_j.csv') %>%
    fread

J = nrow(brand_category)

# prepare data for bayesm
cat(sprintf('date range = [%d, %d]', min(baskets[['t']]), max(baskets[['t']])), fill=TRUE)

# create cluster
sprintf('n_samples = %d (keep = %d)', R, keep) %>% cat(fill=TRUE)

cl = parallel::makeForkCluster(n_cores)
doParallel::registerDoParallel(cl)

# train model by category
# this script assumes that we model 25 categories, each containing 10 products
category_predictions = foreach (category = 0:24) %dopar% {
    if (0) {
        category = 1
    }

    # subset data to category
    category_brands = (0:9)+category*10

    baskets_c =
        baskets %>%
        subset(j %in% category_brands) %>%
        subset(t >= t_train_start-max(avg_windows))

    if (use_discounts) {
        actions_c =
            actions %>%
            subset(j %in% category_brands) %>%
            subset(t >= t_train_start-max(avg_windows))
    }

    # subset data to products with a minimum number of observations
    used_js = baskets_c[, .N, by=j][N>=5, j]
    baskets_c = baskets_c[j %in% used_js]
    baskets_c[, j2 := .GRP, by=j]
    j_map = baskets_c[, list(j, j2)] %>% unique()
    if (use_discounts) {
        actions_c = actions_c[j %in% used_js]
        actions_c = merge(actions_c, j_map, by='j', all.x=TRUE)
        expect_true(sum(is.na(actions_c))==0)
    }
    brand_category_c = merge(j_map, brand_category, 'j', all.x=TRUE)

    # category-level sales
    baskets_c_wide = zeros(I,100)
    baskets_c_wide[baskets_c[, cbind(i+1,t+1)]] = 1
    all(range(baskets_c_wide) == c(0,1)) %>% expect_true

    # build x matrix
    J = nrow(brand_category_c)
    option_none = J+1
    x_default =
        diag(J) %>%
        cbind(brand_category_c[order(j2)][['p_jc']]) %>%
        cbind(matrix(0, J, length(avg_windows))) %>%
        rbind(0) %>%
        set_colnames(c(sprintf('alt_%02d', 1:J), 'price', sprintf('window%02d', avg_windows)))
    if (!use_discounts) {
        x_default = x_default[, colnames(x_default)!='price']
    }

    x_train = x_default[rep(seq.int(nrow(x_default)), T_train), ]
    x_test = x_default[rep(seq.int(nrow(x_default)), T_test), ]
    x_prob_uplift = x_default[rep(seq.int(nrow(x_default)), J+1), ]

    # initialize HBMNL data with outside good purchases and raw x
    dat_train = lapply(1:I, function (i) list('y'=rep(option_none, T_train), 'X'=x_train))
    dat_test = lapply(1:I, function (i) list('y'=rep(option_none, T_test), 'X'=x_test))
    dat_prob_uplift = lapply(
        1:I_prob_uplift,
        function (i) list('y'=rep(option_none, 1), 'X'=x_prob_uplift)
    )

    # update purchase vectors (y)
    cnt_discarded = 0
    for (row in 1:nrow(baskets_c)) {
        baskets_cat_row = baskets_c[row]
        i_row = as.integer(baskets_cat_row[['i']] + 1)      # i + 1
        t_row = as.integer(baskets_cat_row[['t']] + 1)      # t + 1
        # skip data if observation before training window
        if (t_row <= t_train_start) next
        # use data if observation in training or test window
        product_idx = as.integer(baskets_cat_row[['j2']])
        # write to bayesm input list
        # +1 because of t_row is shifted by 1
        if (t_row %between% c(t_train_start+1, t_train_end+1)) {
            dat_train[[i_row]]$y[t_row-t_train_start] = product_idx
        # +1 because of t_row is shifted by 1
        } else if (t_row %between% c(t_test_start+1, t_test_end+1)) {
            dat_test[[i_row]]$y[t_row-t_test_start] = product_idx
        } else {
            cnt_discarded = cnt_discarded + 1
        }
    }
    (cnt_discarded==0) %>% expect_true
    # run 1000 tests
    tmp_test = baskets_c[t<=t_test_end]
    for (nx in 1:1000) {
      test_idx = sample(nrow(tmp_test), 1)
      test_i = tmp_test[test_idx, i]
      test_t = tmp_test[test_idx, t]
      test_j = tmp_test[test_idx, j2]
      if (test_t %between% c(t_train_start, t_train_end)) {
          expect_true(dat_train[[test_i+1]]$y[test_t+1-t_train_start] == test_j)
      } else if (test_t %between% c(t_test_start, t_test_end)) {
          expect_true(dat_test[[test_i+1]]$y[test_t+1-t_test_start] == test_j)
      } else {
          cnt_discarded = cnt_discarded + 1
      }
    }
    rm(tmp_test)
    rm(test_idx)

    # update prices
    if (use_discounts) {
    cnt_discarded = 0
    for (row in 1:nrow(actions_c)) {
        actions_cat_row = actions_c[row]
        i_row = as.integer(actions_cat_row[['i']] + 1)
        t_row = as.integer(actions_cat_row[['t']] + 1)
        prod_idx = as.integer(actions_cat_row[['j2']])
        disc = as.numeric(actions_cat_row[['discount']])

        if (t_row <= t_train_start) next
        if (t_row %between% c(t_train_start+1, t_train_end+1)) {
            dat_train[[i_row]]$X[prod_idx+(t_row-t_train_start-1)*(J+1),J+1] =
                 dat_train[[i_row]]$X[prod_idx+(t_row-t_train_start-1)*(J+1),J+1] * (1-disc)
        } else if (t_row %between% c(t_test_start+1, t_test_end+1)) {
            dat_test[[i_row]]$X[prod_idx+(t_row-t_test_start-1)*(J+1),J+1] =
                dat_test[[i_row]]$X[prod_idx+(t_row-t_test_start-1)*(J+1),J+1] * (1-disc)
        } else {
            cnt_discarded = cnt_discarded + 1
        }
    }
    (cnt_discarded==0) %>% expect_true
    # run 1000 tests
    tmp_test = actions_c[t<=t_test_end]
    for (nx in 1:1000) {
      test_idx = sample(nrow(tmp_test), 1)
      test_i = tmp_test[test_idx, i]
      test_t = tmp_test[test_idx, t]
      test_j = tmp_test[test_idx, j2]
      test_d = brand_category_c[j2==test_j,p_jc] * tmp_test[test_idx, (1-discount)]
      if (test_t %between% c(t_train_start, t_train_end)) {
          expect_true(
              dat_train[[test_i+1]]$X[
                  ((test_t+1)-t_train_start-1)*(J+1) + test_j,
                  'price'
              ] == test_d
          )
      } else if (test_t %between% c(t_test_start, t_test_end)) {
          expect_true(
              dat_test[[test_i+1]]$X[
                  ((test_t+1)-t_test_start-1)*(J+1) + test_j,
                  'price'
              ] == test_d
          )
      } else {
          cnt_discarded = cnt_discarded + 1
      }
      # index logic:
      #   test_t+1         # r-python offset
      #   -t_train_start   # t_train_start offset
      #   -1               # add offset of (J+1) for finished week
    }
    rm(tmp_test)
    rm(test_idx)
    }

    # Update prices for Prob Uplift Calculation
    if (args$u) {
    for (i_iter in 1:I_prob_uplift){
        for (j_iter in 1:J){
            dat_prob_uplift[[i_iter]]$X[j_iter+(j_iter-1)*(J+1), J+1] =
                dat_prob_uplift[[i_iter]]$X[j_iter+(j_iter-1)*(J+1), J+1] * (1-d_prob_uplift)
       }
    }
    }

    # update window variables
    # note here that t_train_start starts with 0...99, and t_iter starts with 1..T_Train
    for (i in 1:I) {
        for (win in avg_windows) {
            # for training data
            for (t_iter in 1:T_train) {
                dat_train[[i]]$X[
                    (1+(J+1)*(t_iter-1)):(J+(J+1)*(t_iter-1)), sprintf('window%02d',win)
                ] = mean(
                    baskets_c_wide[i, (t_train_start+t_iter-1-win+1):(t_train_start+t_iter-2+1)]
                )
            }
            # for testing data
            for (t_iter in 1:T_test){
                dat_test[[i]]$X[
                    (1+(J+1)*(t_iter-1)):(J+(J+1)*(t_iter-1)), sprintf('window%02d',win)
                ] = mean(
                    baskets_c_wide[i, (t_test_start+t_iter-1-win+1):(t_test_start+t_iter-2+1)]
                )
            }
        }
    }
    # run 1000 tests
    tmp_test = baskets_c[t<=t_test_end]
    for (nx in 1:1000) {
      test_idx = sample(nrow(tmp_test), 1)
      test_win = sample(avg_windows,1)
      test_win_name = sprintf('window%02d', test_win)
      test_i = tmp_test[test_idx, i] # python format
      test_t = tmp_test[test_idx, t] # python format
      test_j = tmp_test[test_idx, j2] # category internal product numbering
      # baskets_c_wide already has shifted/mapped t's (+1 for python -- r index)
      test_win_mean = mean(baskets_c_wide[test_i+1, (test_t+1-1-test_win+1):(test_t+1-1)])
      # baskets_c_wide has "r index format"
      # index logic:
      #    test_t+1      # r-python offset
      #    -1            # end window before current week
      #    (test_win+1)  # window end vs. window start
      if (test_t %between% c(t_train_start, t_train_end)) {
          expect_true(
              dat_train[[test_i+1]]$X[
                  (((test_t+1)-t_train_start-1)*(J+1))+test_j,
                  test_win_name
              ]==test_win_mean
          )
      } else if (test_t %between% c(t_test_start, t_test_end)) {
          expect_true(
              dat_test[[test_i+1]]$X[
                  (((test_t+1)-t_test_start-1)*(J+1))+test_j,
                  test_win_name
              ]==test_win_mean
          )
      }
      # index logic:
      #   test_t+1         # r-python offset
      #   -t_train_start   # t_train_start offset
      #   -1               # add offset of (J+1) for finished week
    }

    # update inventory for prob uplift
    for (i in 1:I_prob_uplift) {
        for (win in avg_windows) {
            for (j in 1:(J+1)){
                dat_prob_uplift[[i]]$X[
                    (1+(J+1)*(j-1)):(J+(J+1)*(j-1)),
                    sprintf('window%02d',win)
                ] = mean(
                    baskets_c_wide[i, (t_prob_uplift-win+1):(t_prob_uplift-1+1)]
                )
            }
        }
    }

    # training
    post_index = seq(1 + 0.5 * R / keep, R / keep)
    expect_true(all((post_index %% 1) == 0))
  	hbmnl = rhierMnlRwMixture(
        Data = list(p=J+1, lgtdata=dat_train),
        Prior = list(ncomp=1),
        Mcmc = list(R=R, keep=keep, nprint=R/50)
    )

    variable_names = colnames(dat_train[[1]]$X)

    # prediction
    pred_category = vector('list')
    for (i in 1:I) {
        xi = dat_test[[i]]$X                          # (j,t) x k
        bi = hbmnl$betadraw[i,,post_index]            # k x d
        exp_ui = matrix(exp(xi %*% bi), J+1, )        # j x (t, d)
        probs_draws = t(t(exp_ui)/colSums(exp_ui))    # j x (t * d)
        probs = matrix(rowMeans(matrix(probs_draws, , length(post_index))), J+1)
        all(abs(colSums(probs)-1) < 1e-14) %>% expect_true
    	  pred_category[[i]] = data.table(
            i=i-1,
            j=rep(j_map[['j']], T_test),
            t=rep(t_test_start:t_test_end, each=J),
            coupon_j=999, # this is added for compatibility with pred_prob_uplift
            phat=c(probs[1:J,])
        )
    }
    pred_category_filled =
      CJ(i=0:(I-1), j=category_brands, t=t_test_start:t_test_end) %>%
      merge(rbindlist(pred_category), by=c('i','j','t'), all.x=TRUE)
    pred_category_filled[is.na(phat), phat := 0]
    pred_category_filled[is.na(coupon_j), coupon_j := 999]

    if (args$u) {
        pred_prob_uplift = vector('list')
        for (i in 1:I_prob_uplift) {
          xi = dat_prob_uplift[[i]]$X                   # (j,t) x k
          bi = hbmnl$betadraw[i,,post_index]            # k x d
          exp_ui = matrix(exp(xi %*% bi), J+1, )        # j x (t, d)
          probs_draws = t(t(exp_ui)/colSums(exp_ui))    # j x (t * d)
          probs = matrix(rowMeans(matrix(probs_draws, , length(post_index))), J+1)
          all(abs(colSums(probs)-1) < 1e-14) %>% expect_true
          pred_prob_uplift[[i]] = data.table(
            i=i-1,
            j=rep(j_map[['j']], J+1),
            t=100,
            coupon_j=rep(c(j_map[['j']],999),each=J),
            phat=c(probs[1:J,])
          )
        }

        pred_prob_uplift_filled =
          CJ(i=0:(I_prob_uplift-1), j=category_brands, t=100,coupon_j=c(category_brands,999)) %>%
          merge(rbindlist(pred_prob_uplift), by=c('i','j','coupon_j','t'), all.x=TRUE)
        pred_prob_uplift_filled[is.na(phat)&!(j %in% used_js), phat := 0]

        pred_prob_uplift_no_disc = pred_prob_uplift_filled[coupon_j==999,][,c('i','j','phat')]
        colnames(pred_prob_uplift_no_disc) = c('i','j','phat_no_disc')

        pred_prob_uplift_filled = setDT(pred_prob_uplift_filled)[
            pred_prob_uplift_no_disc,
            on =c("i", "j")
        ][
            is.na(phat),
            phat:= phat_no_disc
        ][,
            phat_no_disc:=NULL
        ][]

        pred_category_filled = rbind(pred_category_filled,pred_prob_uplift_filled)
    }

    pred_category_filled
}

# collect category results
predictions = category_predictions %>% rbindlist

# test probability uplift data
test_0 = predictions[t!=100]
test_0_idx_1 = test_0[, list(i, j, t)]
setkey(test_0_idx_1, i, j, t)
test_0_idx_2 = CJ(i=0:(I-1), j=0:(J-1), t=t_test_start:t_test_end)
setkey(test_0_idx_2, i, j, t)
all.equal(test_0_idx_1, test_0_idx_2) %>% expect_true

# test probability uplift data
if (args$u) {
    test_1 = predictions[coupon_j==999 & t==100]
    test_1_idx_1 = test_1[, list(i, j, t)]
    setkey(test_1_idx_1, i, j, t)
    test_1_idx_2 = CJ(i=0:(I_prob_uplift-1), j=0:(J-1), t=100)
    setkey(test_1_idx_2, i, j, t)
    all.equal(test_1_idx_1, test_1_idx_2) %>% expect_true

    test_2 = predictions[coupon_j!=999 & t==100]
    test_2[, min(abs(j-coupon_j))==0] %>% expect_true
    test_2[, max(abs(j-coupon_j))==9] %>% expect_true
    (nrow(test_2) == (I_prob_uplift * J * 10)) %>% expect_true
    test_2_idx_1 = test_2[, list(i, j, coupon_j, t)]
    setkey(test_2_idx_1, i, j, coupon_j, t)
    test_2_idx_2 = CJ(i=0:(I_prob_uplift-1), j=0:(J-1), coupon_j=0:(J-1), t=100)
    test_2_idx_2[, c_j := j %/% 10]
    test_2_idx_2[, c_j_coupon := coupon_j %/% 10]
    test_2_idx_2 = test_2_idx_2[c_j == c_j_coupon][, list(i,j,coupon_j,t)]
    setkey(test_2_idx_2, i, j, coupon_j, t)
    all.equal(test_2_idx_1, test_2_idx_2) %>% expect_true
}

# the easy part: probability predictions for t in [t_test_start, t_test_end]
test_predictions = predictions[t<100, list(i, j, t, phat)]
test_predictions[, i := as.integer(i)]
test_predictions[, j := as.integer(j)]
test_predictions[, t := as.integer(t)]

# the difficult part: probability for uplift calculations
if (args$u) {
    prob_uplift_filled =
        CJ(i=0:(I_prob_uplift-1), j=0:249, t=100, coupon_j=c(0:249,999)) %>%
        merge(predictions[t==100], by=c('i','j','coupon_j','t'), all.x=TRUE) %>%
        .[, c_j := j %/% 10] %>%
        .[, c_j_coupon := coupon_j %/% 10]
    # cross-category effects are missing
    prob_uplift_filled[(c_j != c_j_coupon) & coupon_j != 999, all(is.na(phat))] %>% expect_true
    # extract probabilities without discounts
    no_discount_predictions = predictions[coupon_j==999 & t==100]
    (nrow(no_discount_predictions) == (I_prob_uplift * J)) %>% expect_true
    # set all cross-category effects to probabilities without discounts
    prob_uplift_filled = merge(
        prob_uplift_filled,
        no_discount_predictions[, list(i, j, t, phat0=phat)],
        by=c('i','j','t'),
        all.x=TRUE
    )
    prob_uplift_filled[
        (c_j != c_j_coupon) & (coupon_j!=999),
        phat := phat0
    ]
    prob_uplift_filled[, sum(is.na(phat))==0] %>% expect_true

    # add price
    prob_uplift_filled = merge(prob_uplift_filled, brand_category[, list(j,p_jc)], by='j')
    prob_uplift_filled[, price_paid := p_jc * (1-d_prob_uplift*(j==coupon_j))]
    prob_uplift_filled[, prob := phat]
    prob_uplift_filled$discount = 'discount'
    prob_uplift_filled[coupon_j==999, discount := 'no discount']
    prob_uplift_filled = prob_uplift_filled[, list(i,j,coupon_j,discount,price_paid,prob)]
    prob_uplift_filled[, i := as.integer(i)]
    prob_uplift_filled[, j := as.integer(j)]
    prob_uplift_filled[, coupon_j := as.integer(coupon_j)]
    prob_uplift_filled[discount == 'no discount', all(coupon_j==999)] %>% expect_true
    prob_uplift_filled[discount == 'no discount', coupon_j := 0]

    tmp_uplift_true = pd$read_parquet(file_uplift_true)
    uplift_true = data.table(
      i=as.integer(tmp_uplift_true$i$values),
      j=as.integer(tmp_uplift_true$j$values),
      coupon_j=as.integer(tmp_uplift_true$coupon_j$values),
      discount=as.character(tmp_uplift_true$discount$values)
    )

    uplift_true[, idx := .I]
    prob_uplift = merge(
        uplift_true,
        prob_uplift_filled,
        by=c('i','j','discount','coupon_j'),
        all.x=TRUE
    )
    prob_uplift[, all(!is.na(prob))] %>% expect_true
    setkey(prob_uplift, idx)
    prob_uplift[, idx := NULL]
    prob_uplift[, all(!is.na(prob))] %>% expect_true
    prob_uplift = prob_uplift[, list(i, j, coupon_j, discount, price_paid, prob)]

    all.equal(
        prob_uplift[, list(i,j,coupon_j,discount)],
        uplift_true[, list(i,j,coupon_j,discount)]
    )
}

# save results
test_predictions_py = r_to_py(test_predictions)
test_predictions_py$to_parquet(file_result, "pyarrow")

if (args$u) {
  prob_uplift_py = r_to_py(prob_uplift)
  prob_uplift_py$to_parquet(file_prob_uplift, "pyarrow")
}

# shut down cluster
parallel::stopCluster(cl)

cat("□", fill=TRUE)

