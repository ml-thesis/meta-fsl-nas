from metanas.metanas.meta_optimizer.agent import NAS_agent


class SAC_RL2_agent(NAS_agent):
    def _init_(self, meta_model, config):
        self.meta_model = meta_model
        self.config = config

    def test_agent(self, env):
        return NotImplementedError

    def act_on_episode(self):
        return NotImplementedError
