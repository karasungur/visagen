import torch

from visagen.training.dfl_module import DFLModule
from visagen.training.optimizers.adabelief import AdaBelief


class TestDFLOptimizerConfig:
    def test_default_optimizer_is_adamw(self):
        # Default should be AdamW
        model = DFLModule()
        optimizers = model.configure_optimizers()

        # In AE mode, returns dict with "optimizer" key
        if isinstance(optimizers, dict):
            optimizer = optimizers["optimizer"]
        else:
            # Should not happen for AE mode default
            optimizer = optimizers[0][0]

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 1e-4
        assert optimizer.defaults["weight_decay"] == 0.01

    def test_explicit_adamw_optimizer(self):
        model = DFLModule(optimizer_type="adamw", learning_rate=5e-5)
        optimizers = model.configure_optimizers()

        if isinstance(optimizers, dict):
            optimizer = optimizers["optimizer"]
        else:
            optimizer = optimizers[0][0]

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 5e-5

    def test_adabelief_optimizer(self):
        # Test AdaBelief instantiation with custom params
        model = DFLModule(
            optimizer_type="adabelief",
            learning_rate=2e-4,
            lr_dropout=0.5,
            lr_cos_period=1000,
            clipnorm=1.0,
        )
        optimizers = model.configure_optimizers()

        if isinstance(optimizers, dict):
            optimizer = optimizers["optimizer"]
        else:
            optimizer = optimizers[0][0]

        assert isinstance(optimizer, AdaBelief)
        assert optimizer.defaults["lr"] == 2e-4
        assert optimizer.defaults["lr_dropout"] == 0.5
        assert optimizer.defaults["lr_cos_period"] == 1000
        assert optimizer.defaults["clipnorm"] == 1.0

    def test_gan_mode_optimizers(self):
        # Test that all optimizers respect the type in GAN mode
        model = DFLModule(optimizer_type="adabelief", gan_power=0.1, gan_mode="vanilla")
        # Initialize necessary components manually since we are not running full init
        # actually init calls _init_gan so they should be there.

        optimizers_tuple = model.configure_optimizers()
        assert isinstance(optimizers_tuple, tuple)

        optimizer_list = optimizers_tuple[0]
        assert len(optimizer_list) == 2  # G and D

        g_opt = optimizer_list[0]
        d_opt = optimizer_list[1]

        assert isinstance(g_opt, AdaBelief)
        assert isinstance(d_opt, AdaBelief)

    def test_temporal_gan_mode_optimizers(self):
        # Test that all optimizers respect the type in Temporal+GAN mode
        model = DFLModule(
            optimizer_type="adabelief",
            gan_power=0.1,
            temporal_enabled=True,
            temporal_power=0.1,
        )

        optimizers_tuple = model.configure_optimizers()
        assert isinstance(optimizers_tuple, tuple)

        optimizer_list = optimizers_tuple[0]
        assert len(optimizer_list) == 3  # G, D, T

        for opt in optimizer_list:
            assert isinstance(opt, AdaBelief)
