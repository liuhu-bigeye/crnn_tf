import os
import pdb
import sys
import re
import copy
import numpy as np
import matplotlib.pyplot as plt

def save(path, ext='png', close=True, verbose=True):
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    #The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),
    # Actually save the figure
    plt.savefig(savepath)
    # Close it
    if close:
        plt.close()
    if verbose:
        print("Done")

if __name__ == '__main__':
    ctc_fname = sys.argv[1]
    em_fname = sys.argv[2]

    re_dev_str = 'dev set.*, losses = \[.*\]'
    re_test_str = 'test set.*, losses = \[.*\]'
    re_train_str = 'Training loss = \[.*\]'

    for phase, fname, c in zip(['ctc', 'em'], [ctc_fname, em_fname], [['r', 'g', 'b'], ['c', 'm', 'k']]):
        losses = []
        loss = []
        with open(fname) as f:
            for line in f.readlines():
                for match in re.findall(re_dev_str, line):
                    if len(loss) > 0:
                        iteration = int(re.findall('Epoch [0-9]*', line)[0].split(' ')[-1])-1
                        loss[-1] = np.array(loss[-1]).mean(axis=0)
                        loss.append(iteration)
                        losses.append(loss)
                    wer = float(re.findall('WER = [0-9\.]*', match)[0].split(' ')[-1])*100
                    match = [wer] + map(float, re.findall('[0-9\.]+', match.split('= ')[-1]))
                    loss = [match, [], []]
                for match in re.findall(re_test_str, line):
                    wer = float(re.findall('WER = [0-9\.]*', match)[0].split(' ')[-1])*100                
                    match = [wer] + map(float, re.findall('[0-9\.]+', match.split('= ')[-1]))
                    loss[1] = match
                for match in re.findall(re_train_str, line):
                    match = map(float, re.findall('[0-9\.]+', match.split('= ')[-1]))
                    # pdb.set_trace()
                    loss[2].append(match)

        iters = [l[-1] for l in losses]
        ctc_dev = [l[0][-3] for l in losses]
        ctc_test = [l[1][-3] for l in losses]
        ctc_train = [l[2][-3] for l in losses]

        best_dev = [l[0][-2] for l in losses]
        best_test = [l[1][-2] for l in losses]
        best_train = [l[2][-2] for l in losses]

        greed_dev = [l[0][-1] for l in losses]
        greed_test = [l[1][-1] for l in losses]
        greed_train = [l[2][-1] for l in losses]

        wer_dev = [l[0][0] for l in losses]
        wer_test = [l[1][0] for l in losses]

        # pdb.set_trace()
        # plt.plot(iters, ctc_train, color=c[0], label='%s: ctc_train'%phase)
        # plt.plot(iters, best_train, color=c[1], label='%s: best_train'%phase)
        # plt.plot(iters, greed_train, color=c[2], label='%s: greed_train'%phase)

        plt.plot(iters, ctc_dev, color=c[0], label='%s: ctc_dev'%phase)
        plt.plot(iters, best_dev, color=c[1], label='%s: best_dev'%phase)
        plt.plot(iters, greed_dev, color=c[2], label='%s: greed_dev'%phase)
        plt.plot(iters, wer_dev, color=c[2], label='%s: WER_dev'%phase)


    plt.ylim([0, 70])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    # handles, labels = plt.get_legend_handles_labels()
    # plt.legend(handles=[ct_line, bt_line, gt_line])

    save('dev', ext='png', close=True)
