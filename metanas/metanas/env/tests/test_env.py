
from metanas.env.core import NasEnv
from metanas.utils.test_config import init_config
from metanas.models.search_cnn import SearchCNNController
import metanas.utils.genotypes as gt


def init_env():

    # Create dummy meta-model and config to test env
    config = init_config()
    meta_model = SearchCNNController(
        16,
        config.init_channels,
        10,
        config.layers,
        n_nodes=config.nodes,
        reduction_layers=config.reduction_layers,
        device_ids=config.gpus,
        PRIMITIVES=gt.PRIMITIVES_FEWSHOT,
        feature_scale_rate=1,
        primitive_space=config.primitives_type,
        weight_regularization=config.darts_regularization,
        use_hierarchical_alphas=config.use_hierarchical_alphas,
        use_pairwise_input_alphas=config.use_pairwise_input_alphas,
        alpha_prune_threshold=config.alpha_prune_threshold,
    )

    env = NasEnv(config, meta_model, test_env=True)

    for i in range(0, 20):
        obs, rew, done, info_dict = env.step(i)

        print(f"step: {i} with reward {rew}")
        print(f"current state: {obs}")
        print(f"done: {done}")
        print(f"info: {info_dict}")


if __name__ == "__main__":
    init_env()
