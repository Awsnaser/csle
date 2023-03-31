from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import geom
from csle_common.metastore.metastore_facade import MetastoreFacade
from csle_common.util.plotting_util import PlottingUtil
from csle_common.dao.system_identification.emulation_statistics import EmulationStatistics
from sklearn.mixture import GaussianMixture
from csle_common.util.export_util import ExportUtil


def plot_hist(statistic: EmulationStatistics, attack_counts: Dict, ips: List[str], condition: str, metric: str) -> None:
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 0.02
    # plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 0.8
    plt.rcParams['axes.linewidth'] = 0.8
    fontsize = 16.5
    labelsize = 15
    # plt.rcParams.update({'font.size': 10})

    ncols= 8
    nrows = 8
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 13))
    col = 0
    row = 0
    print(statistic.conditionals_counts[condition].keys())

    max_val = 0
    for ip in ips:
        x = statistic.conditionals_counts[condition][metric.format(ip)]
        if max(list(x.keys())) > max_val:
            max_val = max(list(x.keys()))
    if max(list(attack_counts.keys())) > max_val:
        max_val = max(list(attack_counts.keys()))

    for i, ip in enumerate(ips):
        x = statistic.conditionals_counts[condition][metric.format(ip)]
        X = []
        X_set = set()
        for val, count in x.items():
            X.append([val])
            X_set.add(val)
        print(ip)
        gmm = GaussianMixture(n_components=2).fit(X)
        mixture_weights = list(gmm.weights_)
        means = list(gmm.means_.tolist())
        covariances = list(gmm.covariances_.tolist())
        sample_space = list(X_set)
        sample_space.sort()
        dists = []
        for weight, mean, covar in zip(mixture_weights, means, covariances):
            dists.append(list(weight * norm.pdf(sample_space, mean, np.sqrt(covar)).ravel()))
        combined_dist = np.zeros(len(sample_space))
        for dist in dists:
            d_arr = np.array(dist)
            combined_dist = combined_dist + d_arr
        combined_distribution = list(combined_dist)
        n, bins, patches = ax[row][col].hist(x, cumulative=False, density=True, bins=10, alpha=1, color="red",
                                   edgecolor='black', linewidth=1, ls="dashdot", label="no intrusion")
        attack_counts_temp = attack_counts.copy()
        for k,v in x.items():
            if k in attack_counts_temp:
                attack_counts_temp[k] = attack_counts_temp[k] + v
            else:
                attack_counts_temp[k] = v
        n, bins, patches = ax[row][col].hist(attack_counts_temp,
                                             cumulative=False, density=True, bins=20, alpha=0.4, color="blue",
                                             edgecolor='black', linewidth=1, ls="dashed", label="intrusion")
        # ax[row][col].plot(sample_space, combined_distribution, 'k--', label='Theoretical', linewidth=2.5)
        ax[row][col].set_title(r"$Z_{\mathbf{O}_{" + str(i+1) + "}}$", fontsize=fontsize)
        # ax[row][col].set_xlabel(r"IDPS alerts weighted by priority", fontsize=fontsize)
        # ax[row][col].set_ylabel(r"$Z_i$", fontsize=fontsize)
        ax[row][col].set_yticks([])
        # ax[row][col].set_xlim(1, max(list(x.keys())))
        ax[row][col].set_xlim(1, max_val)
        xlab = ax[row][col].xaxis.get_label()
        ylab = ax[row][col].yaxis.get_label()
        xlab.set_size(fontsize)
        ylab.set_size(fontsize)
        ax[row][col].spines['right'].set_color((.8, .8, .8))
        ax[row][col].spines['top'].set_color((.8, .8, .8))
        ax[row][col].tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
        ax[row][col].tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)
        if col != 0:
            ax[row][col].set_yticks([])
        if col == 0:
            ax[row][col].set_ylabel(r"probability", fontsize=fontsize)
        if row != nrows-1:
            ax[row][col].set_xticks([])
        else:
            ax[row][col].set_xlabel(r"$\mathcal{O}$", fontsize=fontsize)
        if col == ncols-1:
            col = 0
            row = row + 1
        else:
            col = col + 1
    fig.suptitle(r"Distributions of \# alerts weighted by priority $Z_{\mathbf{O}_i}(\mathbf{O}_i \mid \mathbf{S}^{(\mathrm{D})}_i, \mathbf{A}^{(\mathrm{A})}_{i})$ per node $i \in \mathcal{V}$", fontsize=18)
    handles, labels = ax[0][0].get_legend_handles_labels()
    # handles_2, labels_2 = ax[1][1].get_legend_handles_labels()
    # handles = handles + handles_2
    # labels = labels+labels_2
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.52, 0.0),
               ncol=8, fancybox=True, shadow=False, handletextpad=0.4, labelspacing=0.5, columnspacing=0.65,
               prop={'size': fontsize})
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.2, bottom=0.077)
    plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig("geo_only_2" + ".png", format="png", dpi=600)
    fig.savefig("geo_only_2" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)


if __name__ == '__main__':
    attack_statistic = MetastoreFacade.get_emulation_statistic(id=3)
    print(attack_statistic.conditionals_counts["intrusion"].keys())
    # attack_counts = attack_statistic.conditionals_counts["intrusion"]["alerts_weighted_by_priority"]
    attack_counts = attack_statistic.conditionals_counts["intrusion"]["alerts_weighted_by_priority_16.4.2.3"]
    conditions = ["A:Continue_D:Continue_M:[]", "A:DVWA SQL Injection Exploit_D:Continue_M:[]",
                  "A:FTP dictionary attack for username=pw_D:Continue_M:[]", "A:Network service login_D:Continue_M:[]",
                  "A:Ping Scan_D:Continue_M:[]", "A:SSH dictionary attack for username=pw_D:Continue_M:[]",
                  "A:Sambacry Explolit_D:Continue_M:[]", "A:ShellShock Explolit_D:Continue_M:[]",
                  "A:TCP SYN (Stealth) Scan_D:Continue_M:[]",
                  "A:Telnet dictionary attack for username=pw_D:Continue_M:[]",
    "A:CVE-2015-1427 exploit_D:Continue_M:[]"
                  ]
    print(attack_counts)
    statistic = MetastoreFacade.get_emulation_statistic(id=1)
    metric = "alerts_weighted_by_priority_{}"
    condition = "A:Continue_D:Continue_M:[]"
    ips = [
        '15.13.2.10', '15.13.2.120', '15.13.28.3', '15.13.29.4', '15.13.29.5', '15.13.29.6', '15.13.29.7', '15.13.29.8',
        '15.13.29.9', '15.13.29.110', '15.13.29.11', '15.13.28.12', '15.13.30.13', '15.13.30.14',
        '15.13.30.15', '15.13.30.16', '15.13.30.17', '15.13.30.18', '15.13.30.19', '15.13.30.20',
        '15.13.2.21', '15.13.3.22', '15.13.5.23', '15.13.6.24', '15.13.6.25', '15.13.6.26', '15.13.6.27',
        '15.13.4.28', '15.13.7.29', '15.13.7.30', '15.13.7.31',
        '15.13.7.32', "15.13.8.33", '15.13.10.34', "15.13.8.35", '15.13.10.36', "15.13.9.37",
        '15.13.11.38', "15.13.9.39", '15.13.11.40', '15.13.12.41', '15.13.13.42', '15.13.14.43', '15.13.15.44',
        '15.13.16.45', '15.13.17.46', '15.13.18.47', '15.13.19.48', '15.13.20.49', '15.13.20.50',
        '15.13.21.51', '15.13.21.52', '15.13.22.53', '15.13.22.54', '15.13.23.55', '15.13.23.56',
        '15.13.24.57', '15.13.24.58', '15.13.25.59', '15.13.25.60', '15.13.26.61', '15.13.26.62',
        '15.13.27.63', '15.13.27.64'
    ]
    # plot_hist(statistic, ips=ips, metric=metric, condition=condition, attack_counts=attack_counts)
    ExportUtil.export_emulation_statistics_to_disk_json(
        output_dir="/home/kim/statistics_dataset_31_mar_defender_23_json",
        zip_file_output="/home/kim/statistics_dataset_31_mar_defender_23_json.zip", added_by="Kim Hammar",
        statistics_id=1)
    ExportUtil.export_emulation_statistics_to_disk_json(
        output_dir="/home/kim/statistics_dataset_31_mar_attacker_23_json",
        zip_file_output="/home/kim/statistics_dataset_31_mar_attacker_23_json.zip", added_by="Kim Hammar",
        statistics_id=3)