import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class FeedbackDataset(Dataset):
    def __init__(self, cfg, df, mode, tokenizer):
        assert mode in ['train', 'test']
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.text = df['text'].values
        self.mode = mode
        if mode == 'train':
            self.target = df['discourse_effectiveness']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        inputs = self.tokenizer.encode_plus(
            self.text[item],
            truncation=True,
            add_special_tokens=True,
            max_length=self.cfg.max_len
        )
        samples = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
        }

        if 'token_type_ids' in inputs:
            samples['token_type_ids'] = inputs['token_type_ids']

        if self.mode == 'train':
            samples['target'] = self.target[item]

        return samples


class Collate:
    def __init__(self, tokenizer, istrain=True):
        self.tokenizer = tokenizer
        self.isTrain = istrain

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]

        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in
                                   output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in
                                   output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if self.isTrain:
            output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output


def make_tokenizer_dataset_loader(CFG, df, mode, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    collate_fn = Collate(tokenizer, istrain=True if mode == 'train' else False)
    dataset = FeedbackDataset(CFG, df, mode, tokenizer)
    loader = DataLoader(dataset,
                        batch_size=CFG.batch_size,
                        shuffle=False,
                        collate_fn=collate_fn,
                        # num_workers=CFG.num_workers,
                        pin_memory=False,
                        drop_last=False)
    return tokenizer, dataset, loader
