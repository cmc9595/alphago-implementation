from pathlib import Path
from data.sgf_dataset import iter_sl_samples

zip_dir = Path("./data")

count = 0
for i, s in enumerate(iter_sl_samples(zip_dir, include_pass=False, max_games=3)):
    print(i, s.source, s.x.shape, s.y)
    count += 1
    if i >= 15:
        break

print("DONE. samples:", count)
