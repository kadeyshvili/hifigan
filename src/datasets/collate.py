import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    all_wavs = []
    all_melspecs = []
    max_len_wav = 0
    max_len_spec = 0
    paths = []
    initial_lens = []
    predicted_melspecs = []
    predicted_melspec_init_lens = []
    max_predicted_melspec_init_lens = 0
    for item in dataset_items:
        paths.append(item['path'])
        if 'wav' in item.keys():
            all_wavs.append(item['wav'].squeeze(0))
            all_melspecs.append(item['melspec'])
            max_len_wav = max(len(item['wav'].squeeze(0)), max_len_wav)
            max_len_spec =  max(item['melspec'].shape[-1], max_len_spec)
            initial_lens.append(item['wav'].shape[1])
        elif 'generated_text_melspec' in item.keys():
            predicted_melspecs.append(item['generated_text_melspec'])
            predicted_melspec_init_lens.append(item['generated_text_melspec'].shape[-1])
            max_predicted_melspec_init_lens = max(item['generated_text_melspec'].shape[-1], max_predicted_melspec_init_lens)

    if 'wav' in dataset_items[0].keys():
        result_batch['initial_len'] = initial_lens
        padded_wavs = torch.stack([F.pad(wav, (0, max_len_wav - wav.shape[0]), value=0) for wav in all_wavs])
        padded_specs = torch.stack([F.pad(spec, (0, max_len_spec - spec.shape[-1], 0, 0)) for spec in all_melspecs])
        result_batch['wav'] = padded_wavs.unsqueeze(1)
        result_batch['melspec'] = padded_specs

    else:
        result_batch['predicted_melspec_init_lens'] = predicted_melspec_init_lens
        padded_specs = torch.stack([F.pad(spec, (0, max_predicted_melspec_init_lens - spec.shape[-1], 0, 0)) for spec in predicted_melspecs])
        result_batch['generated_from_text_melspec'] = padded_specs
    result_batch['path'] = paths
    return result_batch