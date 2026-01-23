import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

CH = 8 # 8 channels
WIN = 500 # 500ms

class EMGNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(CH * WIN, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 8)
		)

	def forward(self, dat):
		return self.layers(dat)

file_db = {}
def load_file(fn):
	if fn in file_db:
		return file_db[fn]
	with open(fn) as reader:
		lines = [l.replace("\n", "").split()[1:] for l in reader.readlines()[1:]] 
		wave = np.asarray([l[:-1] for l in lines]).astype(np.float64)
		labels = [int(l[-1]) for l in lines]
		entries = len(lines) - WIN
		print(f"Loading {fn} ({entries} windows, {WIN}ms, {CH} channels)")

		sig = np.zeros((entries, CH, WIN))
		for i in range(entries):
			for j in range(CH):
				sig[i, j] = np.absolute(np.fft.fft(wave[i:i+WIN, j]))

		cats = np.zeros((entries, 8))
		cnt = np.zeros(8)
		for i in range(WIN):
			cnt[labels[i]] += 1	
		cats[i] = cnt.copy()
		for i in range(1, entries):
			cnt[labels[i - 1]] -= 1	
			cnt[labels[i - 1 + WIN]] += 1
			cats[i] = cnt.copy()
		for i in range(entries):
			cats[i] /= WIN
			cats[i] = np.exp(cats[i])
			s = np.sum(cats[i])
			cats[i] /= s

		file_db[fn] = (entries, torch.from_numpy(sig).to(dtype=torch.float32), torch.from_numpy(cats).to(dtype=torch.float32))
	return file_db[fn]

def launch():
	if not torch.cuda.is_available():
		print("CUDA not available")
		exit()

	dev = torch.accelerator.current_accelerator()
	model = EMGNet().to(dev)
	loss = nn.CrossEntropyLoss()
	uin = input("Load saved model (empty for new) >>> ")
	if len(uin) > 0:
		model.load_state_dict(torch.load(uin, weights_only=True))
	opt = torch.optim.SGD(model.parameters(), lr=.001)
	
	while True:
		uin = input("Choose task (train/test/save/quit) >>> ")
		if uin == "quit": break
		elif uin == "save":
			uin = input("File name >>> ")
			torch.save(model.state_dict(), uin)
		elif uin == "train":
			uin = input("File name >>> ")
			entries, sig, cats = load_file(uin)	
			sig = sig.to(dev)
			cats = cats.to(dev)
			lo = int(input("Lower bound >>> "))
			assert lo >= 0 and lo < entries
			hi = int(input("Upper bound >>> "))
			assert hi < entries and hi >= lo
			losses = []
			for i in range(lo, hi + 1):
				mp = model(sig[i].flatten())
				loss_obj = loss(mp, cats[i])
				loss_obj.backward()
				opt.step()
				opt.zero_grad()
				losses.append(loss_obj.item())
			plt.plot(losses)
			plt.show()
		elif uin == "test":
			uin = input("File name >>> ")
			entries, sig, cats = load_file(uin)	
			sig = sig.to(dev)
			cats = cats.to(dev)
			lo = int(input("Lower bound >>> "))
			assert lo >= 0 and lo < entries
			hi = int(input("Upper bound >>> "))
			assert hi < entries and hi >= lo
			with torch.no_grad():
				t_loss = 0
				t_suc = 0
				for i in range(lo, hi + 1):
					mp = model(sig[i].flatten())
					t_loss += loss(mp, cats[i]).item()
					t_suc += (mp.argmax(0) == cats[i]).type(torch.float).sum().item()
				print(f"Test x{hi - lo + 1}: success {int(t_suc)}/{hi - lo + 1} {100 * t_suc / (hi - lo + 1):.2f}% avg={t_loss / (hi - lo + 1):.6f}")

if __name__ == "__main__":
	launch()
