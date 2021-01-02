import tensorflow as tf
import glob
import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats as sp

MAX_HORIZON = 1000000000    

def read_data(tf_dir, tag="return_mean"):

    list_of_data = []

    tf_files = glob.glob(tf_dir + "events.out.tfevents*")

    for tf_file in tf_files:
        done = False
        data = []
        frames = []

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

        data = data[:min(len(frames), len(data))]
        frames = frames[:min(len(frames), len(data))]

        if len(data) != 0:
            list_of_data.append((data, frames))

    list_of_data = sorted(list_of_data, key=(lambda pair: min(pair[1])))

    cleaned_frames = []
    cleaned_data = []
    max_frames = 0
    for (data, frames) in list_of_data:
        for i in range(len(data)):
            if frames[i] < max_frames:
                continue
            else:
                max_frames = frames[i]
                cleaned_frames.append(frames[i])
                cleaned_data.append(data[i])


    # Cumulative sum smoothing
    window_width = 10
    cumsum_vec = np.cumsum(np.insert(cleaned_data, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return cleaned_frames[:len(ma_vec)], ma_vec

# Returns (X_data, (Y_average, Y_error))
def read_data_and_average(tf_dirs, tag="return_mean"):
    X_all = None
    Ys = []

    for tf_dir in tf_dirs:
        X, Y = read_data(tf_dir, tag)

        Ys.append(Y)
        if X_all is None:
            X_all = X
        else:
            if len(X) < len(X_all):
                X_all = X
        # print(tf_dir, len(Y))

    N = min([len(Ys[i]) for i in range(len(Ys))])
    M = len(Ys)
    print(M, N, [len(Ys[i]) for i in range(len(Ys))])
    

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


# # Letter-Env UNTIL_1_3_1_2
A_logs = glob.glob("storage-good/RGCN/*Until*/train/")
B_logs = glob.glob("storage-good/GRU/*Until*/train/")
C_logs = glob.glob("storage-good/LSTM/*Until*/train/")
D_logs = glob.glob("storage-good/Myopic/*Until*/train/")
E_logs = glob.glob("storage-good/GRU-no-progression/*Until*/train/")
F_logs = glob.glob("storage-good/No-LTL/*Until*/train/")
lines = [(A_logs, "GNN+progression"), (B_logs, "GRU+progression"), (C_logs, "LSTM+progression"), (D_logs, "Myopic"), (E_logs, "GRU"), (F_logs, "No LTL")]
MAX_HORIZON = 20000000
title = "Letter-Env: Until_1_3_1_2"


# # Letter-Env EVENTUALLY_1_5_1_4
# A_logs = glob.glob("storage-good/RGCN/*Eventually*/train/")
# B_logs = glob.glob("storage-good/GRU/*Eventually*/train/")
# C_logs = glob.glob("storage-good/LSTM/*Eventually*/train/")
# D_logs = glob.glob("storage-good/Myopic/*Eventually*/train/")
# E_logs = glob.glob("storage-good/GRU-no-progression/*Eventually*/train/")
# F_logs = glob.glob("storage-good/No-LTL/*Eventually*/train/")

# lines = [(A_logs, "GNN+progression"), (B_logs, "GRU+progression"), (C_logs, "LSTM+progression"), (D_logs, "Myopic"), (E_logs, "GRU"), (F_logs, "No LTL")]
# MAX_HORIZON = 20000000
# title = "Letter-Env: Eventually_1_5_1_4"

# #Symbol-Env UNTIL_1_3_1_2

# C_logs = glob.glob("symbol-storage/RGCN/*Until_1_3_1_2*/train/")
# D_logs = glob.glob("symbol-storage/GRU/*Until_1_3_1_2*/train/")

# lines = [(C_logs, "GNN"), (D_logs, "GRU")]
# title = "Symbol Env: Until_1_3_1_2"


# #Symbol-Env EVENTUALLY_1_5_1_4
# A_logs = glob.glob("symbol-storage/RGCN/*Eventually_1_5_1_4*/train/")
# B_logs = glob.glob("symbol-storage/GRU/*Eventually_1_5_1_4*epochs:4*lr:0.003*/train/")

# lines = [(A_logs, "GNN"), (B_logs, "GRU")]
# title = "Symbol Env: Eventually_1_5_1_4"

#=============================================================================================================================================
# ## NOTE: For the transfer results you need to comment out the code at the bottom. 

# # Transfer Until_1_3_1_2
# A1_logs = glob.glob("storage-good/RGCN/*Until_1_3_1_2*/train/")
# A2_logs = glob.glob("transfer-storage/RGCN/*pretrained_Until_1_3_1_2*/train/")
# A3_logs = glob.glob("transfer-storage/RGCN/*pretrained-freeze_ltl_Until_1_3_1_2*bs:64*/train/")

# B1_logs = glob.glob("storage-good/GRU/*Until_1_3_1_2*/train/")
# B2_logs = glob.glob("transfer-storage/GRU/*pretrained_Until_1_3_1_2*/train/")
# B3_logs = glob.glob("transfer-storage/GRU/*pretrained-freeze_ltl_Until_1_3_1_2*/train/")

# lines_A = [(A1_logs, "GNN", "-"), (A2_logs, "GNN+pretraining", "dotted")]
# lines_B = [(B1_logs, "GRU", "-"), (B2_logs, "GRU+pretraining", "dotted")]
# MAX_HORIZON = 10000000

# title = "Transfer - Letter-Env: Until_1_3_1_2"
# plt.ylabel("Discounted return")
# plt.xlabel("Frames")

# for data_line, label, linestyle in lines_A:
#     X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return") #return_mean, average_discounted_return, average_reward_per_step
#     plt.plot(X, Y_avg, linewidth = 1, label=label, linestyle=linestyle, color="b")
#     plt.fill_between(X, Y_lower, Y_upper, alpha=0.2, color="b")

# for data_line, label, linestyle in lines_B:
#     X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return") #return_mean, average_discounted_return, average_reward_per_step
#     plt.plot(X, Y_avg, linewidth = 1, label=label, linestyle=linestyle, color="orange")
#     plt.fill_between(X, Y_lower, Y_upper, alpha=0.2, color="orange")

# plt.legend()
# plt.show()

#=============================================================================================================================================


#  #Transfer EVENTUALLY_1_5_1_4
# A1_logs = glob.glob("storage-good/RGCN/*Eventually_1_5_1_4*/train/")
# A2_logs = glob.glob("transfer-storage/RGCN/*pretrained_Eventually_1_5_1_4*/train/")
# A3_logs = glob.glob("transfer-storage/RGCN/*pretrained-freeze_ltl_Eventually_1_5_1_4*bs:64*/train/")

# B1_logs = glob.glob("storage-good/GRU/*Eventually_1_5_1_4*/train/")
# B2_logs = glob.glob("transfer-storage/GRU/*pretrained_Eventually_1_5_1_4*/train/")
# B3_logs = glob.glob("transfer-storage/GRU/*pretrained-freeze_ltl_Eventually_1_5_1_4*/train/")

# lines_A = [(A1_logs, "GNN", "-"), (A2_logs, "GNN+pretraining", "dotted")]
# lines_B = [(B1_logs, "GRU", "-"), (B2_logs, "GRU+pretraining", "dotted")]
# MAX_HORIZON = 10000000
# title = "Transfer - Letter-Env: Eventually_1_5_1_4"
# plt.ylabel("Discounted return")
# plt.xlabel("Frames")

# for data_line, label, linestyle in lines_A:
#     X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return") #return_mean, average_discounted_return, average_reward_per_step
#     plt.plot(X, Y_avg, linewidth = 1, label=label, linestyle=linestyle, color="b")
#     plt.fill_between(X, Y_lower, Y_upper, alpha=0.2, color="b")

# for data_line, label, linestyle in lines_B:
#     X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return") #return_mean, average_discounted_return, average_reward_per_step
#     plt.plot(X, Y_avg, linewidth = 1, label=label, linestyle=linestyle, color="orange")
#     plt.fill_between(X, Y_lower, Y_upper, alpha=0.2, color="orange")

# plt.legend()
# plt.show()

#=============================================================================================================================================


# # Safety
# A_logs = glob.glob("tmp-storage/GRU-pretrained*bs:2048*fpp:2048*lr:0.0003*ent:0.003*prog:full/train/")
# B_logs = glob.glob("tmp-storage/GRU_*bs:1024*fpp:4096*lr:0.0001*ent:0.001*prog:partial/train/")

# lines = [(A_logs, "Best GNN so far"),(B_logs, "Best Myopic")]


#=========================================
plt.figure(figsize=(12,6))
plt.ylabel("Discounted return")
plt.xlabel("Frames")
plt.title(title)

for data_line, label in lines:
    X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return") #return_mean, average_discounted_return, average_reward_per_step
    plt.plot(X, Y_avg, linewidth = 1, label=label)
    plt.fill_between(X, Y_lower, Y_upper, alpha=0.2)




plt.legend()
# plt.savefig("figs/" + title + ".pdf")
plt.show()

