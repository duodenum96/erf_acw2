from erf_acw2.src import get_commonsubj

subjs = get_commonsubj()

with open("/BICNAS2/ycatal/erf_acw2/erf_acw2/commonsubj.txt", "w") as f:
    for subj in subjs:
        f.write(subj + "\n")
