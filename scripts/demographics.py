from erf_acw2.src import get_commonsubj, subjs
import numpy as np

subjlist = get_commonsubj()
all_subjlist = subjs()
task_bad = np.genfromtxt(
        f"/BICNAS2/ycatal/erf_acw2/erf_acw2/badsubjs_haririhammer.txt",
        dtype="str",
    )

rest_bad = np.genfromtxt(
    "/BICNAS2/ycatal/erf_acw2/erf_acw2/badsubjs_rest.txt", dtype="str"
)

rest_subjlist = np.setdiff1d(all_subjlist, rest_bad)
task_subjlist = np.setdiff1d(all_subjlist, task_bad)

print(f"Number of rest subjects: {len(rest_subjlist)}")
print(f"Number of task subjects: {len(task_subjlist)}")

nsubj = len(subjlist)
print(f"Number of subjects: {nsubj}")
demo_data = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/participants.tsv"

import pandas as pd

demo_df = pd.read_csv(demo_data, sep="\t")

c = 0
for i in subjlist:
    if demo_df[demo_df["participant_id"] == i]["sex"].values[0] == "female":
        c += 1

print(f"Number of females: {c}")

c = 0
for i in rest_subjlist:
    if demo_df[demo_df["participant_id"] == i]["sex"].values[0] == "female":
        c += 1

print(f"Number of females in rest: {c}")

c = 0
for i in task_subjlist:
    if demo_df[demo_df["participant_id"] == i]["sex"].values[0] == "female":
        c += 1

print(f"Number of females in task: {c}")
