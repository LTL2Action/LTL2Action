import tensorflow as tf
import glob
import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats as sp


def read_data(tf_file, tag="return_mean"):
	data = []
	frames = []
	done = False
	for e in tf.train.summary_iterator(tf_file):
		for v in e.summary.value:
			if v.tag == "frames":
				frames.append(v.simple_value)
				if v.simple_value > MAX_HORIZON:
					done = True
			elif v.tag == tag:
				data.append(v.simple_value)
		if done:
			break

	data = np.array(data[:min(len(frames), len(data))])
	frames = np.array(frames[:min(len(frames), len(data))])


	return frames, data

# Returns (X_data, (Y_average, Y_error))
def read_data_and_average(tf_files, tag="return_mean"):
	X_all = None
	Ys = []

	for file in tf_files:
		X, Y = read_data(file, tag)

		Ys.append(Y)
		if X_all is None:
			X_all = X
		else:
			if not np.array_equal(X_all, X):
				print("ERROR: X axis values must all be the same")
				print(X_all[-100:])
				print(X[-100:])
				exit(1)

	N = len(Ys[0])
	M = len(Ys)
	avgs = [0 for i in range(N)]
	lower = [0 for i in range(N)]
	upper = [0 for i in range(N)]

	for n in range(N):
		vals = []
		for m in range(M):
			vals.append(Ys[m][n])

		vals = np.array(vals)
		if M == 1:
			lo = hi = np.mean(vals)
		else:
			lo, hi = sp.t.interval(0.9, len(vals)-1, loc=np.mean(vals), scale=sp.sem(vals))

		avgs[n] = (lo+hi)/2
		lower[n] = lo
		upper[n] = hi

	return X_all, avgs, lower, upper


# Logs for Until_3_3_1_1 
# A_logs = glob.glob("storage/3_3_1_1/RGCN*_UntilTasks_3_3_1_1_Letter-7x7*/train/events.out.tfevents*")
# B_logs = glob.glob("storage/3_3_1_1/rnn-pretrained_UntilTasks_3_3_1_1_Letter-7x7-v3_seed:*_epochs:4_bs:256_fpp:None_dsc:0.99_lr:0.001_ent:0.01_clip:0.2/train/events.out.tfevents*")
# D_logs = glob.glob("storage/3_3_1_1/rnn_UntilTasks_3_3_1_1_Letter-7x7-v3_seed:*_epochs:4_bs:256_fpp:None_dsc:0.99_lr:0.001_ent:0.01_clip:0.2/train/events.out.tfevents*")
# lines = [(A_logs, "GNN Pretrained"), (B_logs, "RNN Pretrained"), (D_logs, "RNN From scratch")]
# MAX_HORIZON = 2900000

# Logs for Until_1_3_1_2
A_logs = glob.glob("storage/1_3_1_2/rnn-pretrained_UntilTasks_1_3_1_2*lr:0.001*/train/events.out.tfevents*")
B_logs = glob.glob("storage/1_3_1_2/rnn_UntilTasks_1_3_1_2*lr:0.001*/train/events.out.tfevents*")
C_logs =  glob.glob("storage/1_3_1_2/rnn-pretrained-unfreeze_ltl_UntilTasks_1_3_1_2*epochs:8*lr:0.001*/train/events.out.tfevents*")
D_logs = glob.glob("storage/1_3_1_2/RGCN_4x16*/train/events.out.tfevents*")

lines = [(B_logs, "From scratch"), (C_logs, "Pretrained (not frozen)"), (D_logs, "RGCN (not frozen)")]
MAX_HORIZON = 19000000
plt.title("Until_1_3_1_2")

# Logs for Sequence_2_4
# A_logs = glob.glob("storage/Sequence_2_4/rnn-pretrained_Sequence_2_4*lr:0.001*/train/events.out.tfevents*")
# B_logs = glob.glob("storage/Sequence_2_4/rnn_Sequence_2_4*lr:0.001*/train/events.out.tfevents*")
# lines = [(A_logs, "Pretrained"), (B_logs, "From scratch")]
# MAX_HORIZON = 700000

# Logs for pretraining 1_3_1_2 GNN vs RNN 
# A_logs = glob.glob("storage/pretraining_1_3_1_2/RGCN*/train/events.out.tfevents*")
# B_logs = glob.glob("storage/pretraining_1_3_1_2/rnn*/train/events.out.tfevents*")
# lines = [(A_logs, "GNN"), (B_logs, "RNN")]
# MAX_HORIZON = 700000

for data_line, label in lines:
	X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line)
	plt.plot(X, Y_avg, linewidth = 1, label=label)
	plt.fill_between(X, Y_lower, Y_upper, alpha=0.2)



plt.legend()
plt.show()