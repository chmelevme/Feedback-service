class CFG:
    num_workers = 1
    model = "microsoft/deberta-large"
    batch_size = 12
    fc_dropout = 0.2
    target_size = 3
    max_len = 512
    seed = 42
    gradient_checkpoint = False
    exp_name = 'Feedback'
    accelerator = 'cpu'
    max_epochs = 3
