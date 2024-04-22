import numpy as np
from csle_common.metastore.metastore_facade import MetastoreFacade
from csle_common.dao.training.experiment_result import ExperimentResult
from csle_common.util.plotting_util import PlottingUtil

if __name__ == '__main__':
    ppo_execution = MetastoreFacade.get_experiment_execution(id=18)

    execution = ppo_execution
    confidence = 0.95
    opt = 1.74

    print(execution.result.all_metrics.keys())
    seeds = list(execution.result.all_metrics.keys())
    returns = []
    runtimes = []
    for seed in seeds:
        if len(execution.result.all_metrics[seed]["running_average_return"]) > 0:
            returns.append(execution.result.all_metrics[seed]["running_average_return"])
        if len(execution.result.all_metrics[seed]["runtime"]) > 0:
            runtimes.append(execution.result.all_metrics[seed]["runtime"])

    return_means = []
    return_cis = []
    runtime_means = []

    for i in range(len(returns[0])):
        return_vals = []
        runtime_vals = []
        for j in range(len(seeds)):
            return_vals.append(returns[j][i])
            runtime_vals.append(runtimes[j][i])
        return_means.append(PlottingUtil.mean_confidence_interval(data=return_vals, confidence=confidence)[0])
        return_cis.append(PlottingUtil.mean_confidence_interval(data=return_vals, confidence=confidence)[1])
        runtime_means.append(PlottingUtil.mean_confidence_interval(data=runtime_vals, confidence=confidence)[0])

    regrets = []
    regret_cis = []
    for i in range(len(returns[0])):
        regret = opt*(i+1) - sum(return_means[0:i])
        regret_ci = sum(return_cis[0:i])
        regrets.append(regret)
        regret_cis.append(regret_ci)

    for i in range(len(regrets)):
        print(f"{runtime_means[i]} {regrets[i]} {regrets[i]-regret_cis[i]/2} "
              f"{regrets[i]+regret_cis[i]/2}")

    # for i in range(len(return_means)):
    #     print(f"{runtime_means[i]} {-return_means[i]} {-return_means[i]-return_cis[i]/1.5} "
    #           f"{-return_means[i]+return_cis[i]/1.5}")
    print(runtime_means)
    # for i in range(len(runtime_means)):
    #     print(runtime_means[i])



