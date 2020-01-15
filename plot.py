import ast
import matplotlib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
from ast import literal_eval
from collections import OrderedDict

matplotlib.use("Agg")

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def get_datasets(logdir, condition=None, ignore_distr=False):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []

    exp_name = None
    for root, xx, files in os.walk(logdir):

        if 'progress.txt' in files:

            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except BaseException:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            exp_data = pd.read_table(os.path.join(root, 'progress.txt'))

            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in \
                exp_data else 'AverageEpRet'
            exp_data.rename(
                inplace=True, columns={
                    'TotalEnvInteracts': 'Iteration'})
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(
                len(exp_data.columns),
                'Average Cumulative Reward', exp_data[performance])
            datasets.append(exp_data)

            if 'distributions.txt' in files and not ignore_distr:
                exp_data_distr = pd.read_table(
                    os.path.join(root, 'distributions.txt'))
                return datasets, exp_data_distr

    if ignore_distr:
        return datasets
    else:
        return datasets, []


def get_all_datasets(all_logdirs,
                     legend=None,
                     select=None,
                     exclude=None,
                     ignore_distr=False):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """

    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == '/':
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)

            def fulldir(x):
                return osp.join(basedir, x)

            prefix = logdir.split('/')[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])
    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [
            log for log in logdirs if all(not (x in log) for x in exclude)
        ]

    # Verify logdirs
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:

        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg, ignore_distr=ignore_distr)
    else:

        for log in logdirs:
            data += get_datasets(log, ignore_distr=ignore_distr)
    return data


def make_separate_plots(all_logdirs,
                        legend=None,
                        xaxis=None,
                        values=None,
                        count=False,
                        font_scale=1.5,
                        smooth=1,
                        select=None,
                        exclude=None,
                        estimator='mean',
                        per_seed_comp=False):

    phases = os.listdir(all_logdirs[0])

    if ('training' in phases):

        all_training_logdirs = all_logdirs[0] + '/training/'
        exp_dir = os.listdir(all_training_logdirs)

        for exp in exp_dir:

            all_training_exps = all_training_logdirs + exp

            train_data, distr_data = get_all_datasets([all_training_exps],
                                                      legend, select, exclude)

            timesteps = train_data[0]['Epoch'].values
            average_returns = train_data[0]['AverageEpRet'].values

            plt.figure()
            plt.plot(timesteps, average_returns)
            plt.title('Average Rewards')
            plt.xlabel('Epoch')
            plt.ylabel('Average rewards per epoch')
            plotname = all_training_exps + '/' + 'AverageTrainRewards.png'
            plt.savefig(plotname)


def make_comparison_plots(exp_dirs):

    curr_dir = os.getcwd()
    colors = ['b', 'r', 'orange', 'green', 'navy', 'm', 'c']

    for idx, exp in enumerate(exp_dirs):
        print(exp, idx)
        exp_dir = curr_dir + '/' + exp

        exp_files = os.listdir(exp_dir)

        if 'eval.log' in exp_files:
            # read the episode_reward
            eps_rewards = []
            eval_file = exp_dir + '/' + 'eval.log'
            with open(eval_file) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    print("Line {}: {}".format(cnt, line.strip()))
                    eps_reward = ast.literal_eval(line)['episode_reward']
                    eps_rewards.append(eps_reward)

                    line = fp.readline()

                    cnt += 1
            plt.plot(eps_rewards, label=exp, color=colors[idx])

        plt.legend()
        plt.title('Average Rewards')
        plt.xlabel('Eval Epoch')
        plt.ylabel('Average rewards per epoch')
        plt.savefig('results/comparisons.png')


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--exp_list', nargs='+',
                        help='<Required> Set flag', required=True)

    # parser.add_argument('logdir', nargs='*')
    # parser.add_argument('--srcdir', nargs='*')
    # parser.add_argument('--legend', '-l', nargs='*')
    # parser.add_argument('--xaxis', '-x', default='Iteration')
    # parser.add_argument(
    #     '--value',
    #     '-y',
    #     default='Average Cumulative Reward',
    #     nargs='*')
    # parser.add_argument('--count', action='store_true')
    # parser.add_argument('--smooth', '-s', type=int, default=1)
    # parser.add_argument('--select', nargs='*')
    # parser.add_argument('--exclude', nargs='*')
    # parser.add_argument('--est', default='mean')
    parser.add_argument('--style', default='compare')
    # parser.add_argument('--root', default='data')
    # parser.add_argument('--fixed_mean', action='store_true')
    # parser.add_argument('--vary_mean', action='store_true')
    args = parser.parse_args()

    if args.style == 'separate':

        make_separate_plots(
            args.logdir,
            args.legend,
            args.xaxis,
            args.value,
            args.count,
            smooth=args.smooth,
            select=args.select,
            exclude=args.exclude,
            estimator=args.est)
    elif args.style == 'compare':

        make_comparison_plots(args.exp_list)

    else:
        raise ValueError('Invalid plot style')


if __name__ == "__main__":
    main()
