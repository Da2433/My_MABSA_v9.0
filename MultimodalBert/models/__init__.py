from MultimodalBert.models.region_bert_classification import RegionBertClassification

def build_model(args):
    task_name = args.task_name
    if task_name == 'MABSA':
        model = RegionBertClassification(args)
    else:
        raise KeyError('Unknown task type...')

    model = model.cuda() if args.is_cuda else model

    return model





