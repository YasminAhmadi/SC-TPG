from pathlib import Path
import numpy as np

data_dir = Path(r"C:\linux_project\SwarmTPG_clean\runs\DefeatZerglingsAndBanelings_WM_RSwarm_LocalMem\wm_data")
files = sorted(data_dir.glob("wm_ep*_gen*.npz"))
print("num_files:", len(files))

n_with_one = 0
total_ones = 0
last_is_one = 0
examples = []

for p in files:
    d = np.load(p, allow_pickle=False)
    if "dones" not in d.files:
        continue
    dones = d["dones"].astype(np.int32)
    s = int(dones.sum())
    if s > 0:
        n_with_one += 1
        total_ones += s
        if int(dones[-1]) == 1:
            last_is_one += 1
        if len(examples) < 10:
            idxs = np.where(dones == 1)[0]
            examples.append((p.name, len(dones), s, int(dones[-1]), idxs[:10].tolist()))

print("files_with_done1:", n_with_one)
print("total_done_ones:", total_ones)
print("done1_at_last_frame:", last_is_one)
print("examples(name, T, sum, last, first_idxs):")
for e in examples:
    print("  ", e)
