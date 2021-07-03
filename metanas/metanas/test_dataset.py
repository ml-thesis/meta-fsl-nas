from metanas.tasks.torchmeta_loader import MixedOmniglotTripleMNISTFewShot


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


config = Namespace(k=20, data_path='/home/rob/Git/meta-fsl-nas/metanas/data',
                   download=True, q=1, n=5, n_train=15,
                   meta_batch_size=10, test_meta_batch_size=10,
                   workers=1, batch_size=20, batch_size_test=10,
                   seed=42, use_vinyals_split=True)

task_distribution = MixedOmniglotTripleMNISTFewShot(config, download=True)


task_distribution.sample_meta_train()
