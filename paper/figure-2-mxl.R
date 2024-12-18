# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

# libraries
suppressMessages({
  library(magrittr)
  library(data.table)
  library(foreach)
  library(doParallel)
  library(bayesm)
  library(glue)
  library(arrow)
  library(pracma)
  library(yaml)
  library(testthat)
  library(reticulate)
})

# python integration
path_virtualenv = Sys.getenv("PATH_VIRTUALENV")
sprintf("path_virtualenv = %s", path_virtualenv) %>% cat(fill=TRUE)
use_virtualenv(path_virtualenv, required=TRUE)
pd = import("pandas", "pd")
pq = import("pyarrow.parquet", "pq")

# GLOBAL DEFINITIONS
config = read_yaml("configs/config.yaml")
path_data =
    config[['path_data']] %>%
    sub(pattern="\\$HOME", replacement=Sys.getenv("HOME"), x=.) %>%
    paste0("_009/figure-2")
sprintf("path_data = %s", path_data) %>% cat(fill=TRUE)
# skip figure 2 if outptut for `figure-2-data.py` does not exist
# equal to data set 9 is not available or simulation config does not match paper
if (!file.exists(glue("{path_data}/pred_lightgbm.parquet"))) {
  q("no")
}
t_train_start = 60
t_train_end = 89
t_test_start = 90
t_test_end = 99
T_train = t_train_end-t_train_start+1
T_test = t_test_end-t_test_start+1
avg_windows = c(1, 3, 5, 15, 30)
R = 50000
keep = 20
I_test = 2000
Imodel = 2000
I = max(I_test, Imodel)

set.seed(1)
n_cores = config[["n_cores"]][["mxl"]]

file_result = glue("{path_data}/pred_mxl.parquet")

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

brand_category =
    file.path(path_data, 'data_j.csv') %>%
    fread

J = nrow(brand_category)

# create cluster
cl = parallel::makeForkCluster(n_cores)
doParallel::registerDoParallel(cl)

category_predictions = foreach (category = 0:24) %dopar% {
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

    x_train = x_default[rep(seq.int(nrow(x_default)), T_train), ]
    x_test = x_default[rep(seq.int(nrow(x_default)), T_test), ]
    x_prob_uplift = x_default[rep(seq.int(nrow(x_default)), J+1), ]

    # initialize HBMNL data with outside good purchases and raw x
    dat_train = lapply(1:I, function (i) list('y'=rep(option_none, T_train), 'X'=x_train))
    dat_test = lapply(1:I, function (i) list('y'=rep(option_none, T_test), 'X'=x_test))

    # update purchase vectors (y)
    cnt_discarded = 0
    for (row in 1:nrow(baskets_c)) {
        baskets_cat_row = baskets_c[row]
        i_row = as.integer(baskets_cat_row[['i']] + 1)
        t_row = as.integer(baskets_cat_row[['t']] + 1)
        # skip data if observation before training window
        if (t_row <= t_train_start) next
        # use data if observation in training or test window
        product_idx = as.integer(baskets_cat_row[['j2']])
        # write to bayesm input list
        if (t_row %between% c(t_train_start+1, t_train_end+1)) {
            dat_train[[i_row]]$y[t_row-t_train_start] = product_idx
        } else if (t_row %between% c(t_test_start+1, t_test_end+1)) {
            dat_test[[i_row]]$y[t_row-t_test_start] = product_idx
        } else {
            cnt_discarded = cnt_discarded + 1
        }
    }
    (cnt_discarded==0) %>% expect_true

    # update prices
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
            print(t_row)
            cnt_discarded = cnt_discarded + 1
        }
    }
    (cnt_discarded==0) %>% expect_true

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
    rm(baskets_c_wide)

    # training
    n_models = I %/% Imodel
    post_index = seq(1 + 0.5 * R / keep, R / keep)
    if (n_models == 1) {
      	hbmnl = rhierMnlRwMixture(
            Data = list(p=J+1, lgtdata=dat_train),
            Prior = list(ncomp=1),
            Mcmc = list(R=R, keep=keep, nprint=R/50)
        )
        betadraws = hbmnl$betadraw[,,post_index]
    } else {
        expect_true(I %% Imodel == 0)
        for (nm in 1:n_models) {
            idx_is = (1+(nm-1)*Imodel) : (nm*Imodel)
            dat_train_nm = dat_train[idx_is]
            missing_js = setdiff(1:(J+1), sort(unique(unlist(lapply(dat_train_nm, '[[', 'y')))))
            if (length(missing_js)>0) {
                tmp = dat_train_nm[[1]]
                tmp$y[sample(length(tmp$y), length(missing_js)*3, replace=FALSE)] = rep(missing_js, 4)
                length(dat_train_nm)
                dat_train_nm[[length(dat_train_nm)+1]] = tmp
            }
          	hbmnl_nm = rhierMnlRwMixture(
                Data = list(p=J+1, lgtdata=dat_train_nm),
                Prior = list(ncomp=1),
                Mcmc = list(R=R, keep=keep, nprint=R/50)
            )
            #variable_names = colnames(dat_train[[1]]$X)
            #hbmnl_draws_nm = reshapeBayesmDraws(hbmnl_nm, post_index, variable_names)
            if (nm == 1) {
              dim_out = dim(hbmnl_nm$betadraw[,,post_index])
              dim_out[1] = I
              betadraws = array(0, dim_out)
            }
            betadraws[idx_is,,] = hbmnl_nm$betadraw[1:length(idx_is),,post_index]
        }
    }

    # prediction
    pred_category = vector('list')
    for (i in 1:I_test) {
        xi = dat_test[[i]]$X                          # (j,t) x k
        bi = betadraws[i,,]                           # k x d
        exp_ui = matrix(exp(xi %*% bi), J+1, )        # j x (t, d)
        probs_draws = t(t(exp_ui)/colSums(exp_ui))    # j x (t * d)
        probs = matrix(rowMeans(matrix(probs_draws, , length(post_index))), J+1)
        all(abs(colSums(probs)-1) < 1e-14) %>% expect_true
    	  pred_category[[i]] = data.table(
            i=i-1,
            j=rep(j_map[['j']], T_test),
            t=rep(t_test_start:t_test_end, each=J),
            phat=c(probs[1:J,])
        )
    }
    pred_category_filled =
      CJ(i=0:(I_test-1), j=category_brands, t=t_test_start:t_test_end) %>%
      merge(rbindlist(pred_category), by=c('i','j','t'), all.x=TRUE)
    pred_category_filled[is.na(phat), phat := 0]

    pred_category_filled
}

# collect category results
predictions = category_predictions %>% rbindlist

# the easy part: probability predictions for t in [t_test_start, t_test_end]
test_predictions = predictions[t<100, list(i, j, t, phat)]
test_predictions[, i := as.integer(i)]
test_predictions[, j := as.integer(j)]
test_predictions[, t := as.integer(t)]

# save results
test_predictions_py = r_to_py(test_predictions)
test_predictions_py$to_parquet(file_result, "pyarrow")

# shut down cluster
parallel::stopCluster(cl)

