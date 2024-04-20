from csle_common.metastore.metastore_facade import MetastoreFacade
from csle_common.dao.training.experiment_result import ExperimentResult

if __name__ == '__main__':
    execution = MetastoreFacade.get_experiment_execution(id=1)
    execution.to_json_file("/home/kim/cpomcp_journal_ppo_case_study_1_20_april.json")