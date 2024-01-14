import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import json

plt.rcParams.update({'font.size': 15})

colors_map = {
    'QPLEX': '#F0E442',
    'QMIX': '#4169E1',
    'VDN': '#20B2AA',
    'QTRAN': '#CC79A7',
    'SHAQ': '#D8A014',
    'OW-QMIX': '#4B0082',
    'CW-QMIX': '#9D4C26',
    'CDS': '#666666',
    'MIXRTs': '#006400',
    'QNAM (Ours)': 'red',
}

_term = 'win_rates'    # win_rates or episode_rewards
algs = ['vdn', 'qmix', 'qtran', 'qplex', 'ow_qmix', 'cw_qmix', 'cds', 'shaq']#, 'mixrts', 'qnam'
q_tree_depth = 3
mix_q_tree_depth = 3
beta = 0

def get_num(map):
    load_num = 200
    if smac_map == '3m':
        load_num = 100
    if smac_map == '2s3z':
        load_num = 150
    if smac_map == '6h_vs_8z' or smac_map == '3s5z_vs_3s6z' or smac_map == 'corridor':
        load_num = 500
    return load_num
nums = 5


def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window
    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out


def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform one-sided (causal) EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]
    Arguments:
    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds
    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]
    n: int                - number of points in new x grid
    decay_steps: float    - EMA decay factor, expressed in new x grid steps.
    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN
    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid
    '''

    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
    assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))


    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0 # last unused old index
    sum_y = 0.
    count_y = 0.
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(- 1. / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            if luoi >= len(xolds):
                break
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(- (xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan

    return xnews, ys, count_ys

def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform symmetric EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]
    Arguments:
    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds
    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]
    n: int                - number of points in new x grid
    decay_steps: float    - EMA decay factor, expressed in new x grid steps.
    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN
    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid
    '''
    xs, ys1, count_ys1 = one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0)
    _,  ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold=0)
    ys2 = ys2[::-1]
    count_ys2 = count_ys2[::-1]
    count_ys = count_ys1 + count_ys2
    ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    ys[count_ys < low_counts_threshold] = np.nan
    return xs, ys, count_ys


def read_file(smac_map, load_num):
    files = {}
    for alg_name in algs:
        term_files = []
        for num in range(1, nums+1):

            file_path = '../results/sacred/'+smac_map+'/'+alg_name+'/'+str(num)+'/info.json'
            print(file_path)
            f = open(file_path, 'r', encoding='utf-8')
            result_data = json.load(f)['test_battle_won_mean']
            result_data = np.array(result_data)[:load_num]
            term_files.append(result_data)
            print(len(result_data))

        term_files = np.array(term_files).T.tolist()
        files[alg_name]=term_files
    return files

def plt_term_mean(smac_map, load_num, files, i, j):
    coll = files
    # plt.figure(figsize=(8, 6), dpi=200)
    ax = plt.subplot(Grid[i, j])

    for alg_name in algs:
        label = None
        if alg_name == 'qmix':
            label = 'QMIX'
        elif alg_name == 'qtran':
            label = 'QTRAN'
        elif alg_name == 'qplex':
            label = 'QPLEX'
        elif alg_name == 'mixrts':
            label = 'MIXRTs'
        elif alg_name == 'ow_qmix':
            label = 'OW-QMIX'
        elif alg_name == 'cw_qmix':
            label = 'CW-QMIX'
        elif alg_name == 'vdn':
            label = 'VDN'
        elif alg_name == 'shaq':
            label = 'SHAQ'
        elif alg_name == 'cds':
            label = 'CDS'
        elif alg_name == 'qnam':
            label = 'QNAM (Ours)'
        # logger = Logger(exp_name=alg_name, env_name=smac_map,)

        mean_values = []
        max_values = []
        min_values = []

        for k, val in enumerate(coll[alg_name]):
            mean = sum(val) / len(val)
            mean_values.append(mean)
            variance = np.std(val)#/(np.sqrt(len(val)))

            max_values.append(mean + variance)
            min_values.append(mean - variance)

        print(label)
        max_idx = np.argmax(mean_values)
        min_idx = np.argmin(mean_values)
        x = np.arange(len(coll[alg_name]))
        y = smooth(mean_values, radius=2)*100
        min_values = smooth(min_values, radius=3)*100
        max_values = smooth(max_values, radius=3)*100
        # x, y, counts = symmetric_ema(x, y, x[0], x[-1], resample, decay_steps=smooth_step)

        plt.plot(x, y, linewidth=2.0, label=label, color=colors_map[label])
        plt.fill_between(np.arange(len(coll[alg_name])), min_values, max_values,
                         color=colors.to_rgba(colors_map[label], alpha=0.15))
    plt.ylim(-2, 101)
    plt.xlim(-2, load_num+1)
    plt.yticks((range(0, 101, 20)))
    if load_num==200:
        plt.xticks((range(0, len(coll[alg_name]) + 3, 50)), ("0", "0.50", "1.00", "1.50", "2.00"))

    if load_num==500:
        plt.xticks((range(0, len(coll[alg_name])+1, 100)), ("0", "1.00", "2.00", "3.00", "4.00", "5.00"))
        # plt.yticks((range(0, 52, 10)))
        # plt.ylim(-2, 52)
        plt.ylabel('Test Win Rate %', labelpad=-0.5)

    # plt.xlabel('Steps '+r'${\times }$ 10K')
    plt.xlabel('T (mil)')
    plt.ylabel('Test Win Rate %', labelpad=-6.5)
    plt.rcParams.update({'font.size': 15})
    plt.title(smac_map)
    # plt.legend(framealpha=0.3)#,  loc='lower right'
    return ax


if __name__ == '__main__':
    # smac_maps = ['8m','2s_vs_1sc', '8m_vs_9m', '2c_vs_64zg', '5m_vs_6m', '3s_vs_5z', 'MMM2', '6h_vs_8z', '3s5z_vs_3s6z']
    smac_maps = ['8m','2s3z','3s5z','2s_vs_1sc', '8m_vs_9m', '2c_vs_64zg', '5m_vs_6m', '3s_vs_5z', 'MMM2', '6h_vs_8z', '3s5z_vs_3s6z']
    # smac_maps = ['5m_vs_6m']
    ax = plt.figure(figsize=(16, 12), dpi=400)
    Grid = plt.GridSpec(4, 3, wspace=0.2, hspace=0.4)
    # Grid = plt.GridSpec(3, 3, wspace=0.2, hspace=0.6)
    plt.rcParams.update({'font.size': 15})

    for i, smac_map in enumerate(smac_maps):
        load_num = get_num(smac_map)
        files = read_file(smac_map, load_num)
        ax = plt_term_mean(smac_map,load_num, files, int(i/3), int(i%3))
        ax.grid(True, alpha=0.3, linestyle='-.')

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, ncol=5, bbox_to_anchor=(0.5, 4.3))
    plt.savefig('./overview_results.pdf',bbox_inches='tight')
    # plt.show()