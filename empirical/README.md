# Empirical Analyses

Readme for the empirical analyses in

> Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable
> Deep-Learning Model." _Management Science_ (forthcoming).

The data for the empirical application is under NDA. The implementations of the DNN
product choice model and the baselines are identical to the simulation study. The current folder contains scripts for data preprocessing and the analyses to demonstrate key decisions in data pruning.

## Code

```
.
├── README.md            # this file
├── preprocess-data.py   # produces teh data (the output is mimicking the synthetic data)
├── table_5.py           # produces the summary statistics
├── table_6.py           # produces the loss benchmarking
└── table_7.R            # produces the loss regression analysis
```

The script `preprocess-data.py` serves the following purposes:
- build new consumer IDs (0, ..., # of users), mimicking the synthetic data
- map products to brands
- aggregate small products to `other` (by category; to make training HB-MNL possible)
- aggregate the product embedding
- drop consumers with less than 15 shopping trips

For more details, see Section 6 in the paper or in-line comments in the script.

After running `preprocess-data.py` the data is formatted just like the synthetic data,
allowing us to use the same code for the synthetic and empirical data. The only difference
is the script for training the HB-MNL that is generalized to deal with varying number of
brands per category. The outputs for the five models (`DNN`, `LightGBM`,
`BinaryLogit`, `HB-MNL`, `PurchaseFrequency`) has the same structure as the results for
the synthetic data:

- `i`: consumer id, from 0 to `I`
- `j`: product ID, from 0 to `J`
- `t`: week number
- `phat`: predicted probability

The remaining scripts produce the tables presented in the paper.


## Data

The empirical analysis is based on the following data:

- `brand_master`: Product-brand map (used for data aggregation to the brand level).
  Contains the fields
    - `j`: product ID
    - `brand`: brand ID
- `data_j`: Product-category structure (similar to `data_250_j_v3.csv`), a map between
  product IDs and category IDs (according to retailer's category definition). Contains the
  fields
    - `j`: product ID
    - `category`: category ID
- `baskets_raw`: Purchase histories. Contains the fields
    - `user`: consumer ID (mapped to `i`)
    - `t`: week number
    - `j`: product ID
- `actions_raw`: Action histories (retailer coupons). Contains the fields
    - `user`: consumer ID (mapped to `i`)
    - `t`: week number
    - `j`: product ID
    - discount (percentage points)
- `w2v_embedding`: Product embeddings using the Product2Vec model (following Gabel et al.
  (2019). P2V-MAP: mapping market structures for large retail assortments. _Journal of
  Marketing Research_, 56(4), pp.557-580), pre-trained on market basket data. The embedding
  is aggregated in `preprocess-data.py`.

