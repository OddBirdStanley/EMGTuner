import numpy as np
import tkinter
from collections import Counter

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 

def launch():
	data_sig = None
	data_label = None

	offset = 0
	size = 1000

	root = tkinter.Tk()
	root.wm_title("Neuro FFT")

	sig_fig = Figure()
	sig_ax = sig_fig.subplots(3, 3)
	sig_plot = FigureCanvasTkAgg(sig_fig, master=root)
	sig_plot.draw()
	def draw_sig():
		for i in range(8):
			sig_ax[i // 3, i % 3].clear()
			sig_ax[i // 3, i % 3].plot(np.fft.fft(data_sig[:, i][offset:offset+size]))
		sig_ax[2, 2].clear()
		cnt = dict(Counter(data_label[offset:offset+size])).items()
		sig_ax[2, 2].pie([i[1] for i in cnt], labels=[i[0] for i in cnt])
		for i in range(8):
			sig_ax[i // 3, i % 3].set_title(f"Channel {i+1}")
		sig_ax[2, 2].set_title("Labels")
		sig_plot.draw()

	controls = tkinter.Frame(master=root)

	def offset_event(e):
		nonlocal offset
		offset = offset_slide.get()
		draw_sig()
	offset_slide = tkinter.Scale(master=controls, label="Offset", from_=0, to_=0, command=offset_event, state="disabled", orient="horizontal", length=500)

	def size_event(e):
		nonlocal size
		size = size_slide.get()
		draw_sig()
	size_slide = tkinter.Scale(master=controls, label="Size", from_=1, to_=1000, command=size_event, state="disabled", orient="horizontal", length=500)

	def load_event():
		nonlocal data_sig, data_label, offset, size
		active_file = tkinter.filedialog.askopenfile()
		if active_file is not None:
			lines = [[float(i) for i in l.replace("\n", "").split("\t")] for l in active_file.readlines()[1:]]
			arr = np.asarray(lines)
			data_sig = arr[:, 1:-1]
			data_label = arr[:, -1]
			size = min(size, len(data_sig))	
			offset = min(len(data_sig)-size, offset)
			offset_slide.configure(to_=len(data_sig)-size)
			size_slide.configure(to_=len(data_sig))
			offset_slide.set(0)
			size_slide.set(size)
			offset_slide.configure(state="normal")
			size_slide.configure(state="normal")
			draw_sig()
	load = tkinter.Button(master=controls, text="Load", command=load_event)

	load.pack(side=tkinter.LEFT)
	offset_slide.pack(side=tkinter.LEFT)
	size_slide.pack(side=tkinter.LEFT)
	controls.pack(side=tkinter.TOP)
	sig_plot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
	tkinter.mainloop()

if __name__ == "__main__":
	launch()
