# import warnings

# import hydra
# import torch
# from hydra.utils import instantiate
# from omegaconf import OmegaConf

# from src.datasets.data_utils import get_dataloaders
# from src.trainer import Trainer
# from src.utils.init_utils import set_random_seed, setup_saving_and_logging
# from src.utils.io_utils import ROOT_PATH


# from speechbrain.inference.TTS import Tacotron2



# warnings.filterwarnings("ignore", category=UserWarning)


# @hydra.main(version_base=None, config_path="src/configs", config_name="syntesize")
# def main(config):
#     set_random_seed(config.inferencer.seed)

#     project_config = OmegaConf.to_container(config)

#     if config.inferencer.device == "auto":
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#     else:
#         device = config.inferencer.device

#     # setup text_encoder

#     # setup data_loader instances
#     # batch_transforms should be put on device
#     dataloaders, batch_transforms = get_dataloaders(config, device)

#     # build model architecture, then print to console
#     model = instantiate(config.model).to(device)

#     # get function handles of loss and metrics

#     metrics = {"inference": []}
#     for metric_config in config.metrics.get("inference", []):
#         # use text_encoder in metrics
#         metrics["inference"].append(
#             instantiate(metric_config)
#         )

#     # epoch_len = number of iterations for iteration-based training
#     # epoch_len = None or len(dataloader) for epoch-based training

#     tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")

#     # Running the TTS    
#     mel_output, _, _ = tacotron2.encode_text("Mary had a little lamb")
    
#     print(mel_output)
#     save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
#     save_path.mkdir(exist_ok=True, parents=True)
#     # epoch_len = config.trainer.get("epoch_len")

#     # trainer = Trainer(
#     #     model=model,
#     #     criterion=loss_function,
#     #     metrics=metrics,
#     #     gen_optimizer=gen_optimizer,
#     #     disc_optimizer=disc_optimizer,
#     #     gen_lr_scheduler=gen_lr_scheduler,
#     #     disc_lr_scheduler=disc_lr_scheduler,
#     #     config=config,
#     #     device=device,
#     #     dataloaders=dataloaders,
#     #     epoch_len=epoch_len,
#     #     logger=logger,
#     #     writer=writer,
#     #     batch_transforms=batch_transforms,
#     #     skip_oom=config.trainer.get("skip_oom", True),
#     # )

#     # trainer.train()


# if __name__ == "__main__":
#     main()

import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH
from speechbrain.inference.TTS import Tacotron2


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="syntesize")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup text_encoder

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    print(model)

    # get metrics
    metrics = {"inference": []}
    for metric_config in config.metrics.get("inference", []):
        # use text_encoder in metrics
        metrics["inference"].append(
            instantiate(metric_config)
        )

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)
    # tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
    # mel_output, _, _ = tacotron2.encode_text("Mary had a little lamb")
    # print(mel_output)



    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
