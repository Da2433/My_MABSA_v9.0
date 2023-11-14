import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utilss.metrics_utils import macro_f1
from utilss.modal_utils import save_model, load_model, load_avg_model_state

from transformers import AdamW, get_linear_schedule_with_warmup
from MultimodalBert.models import build_model
from MultimodalBert.dataset.multimodal_data_module import MultimodalDataModule

import average_checkpoint

class MultimodalTrainer(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.is_cuda = args.is_cuda
        self.patience = args.patience
        self.datamodule = MultimodalDataModule(args, logger)
        self.model = build_model(args)
        self.model = self.model.to(self.args.device) if self.is_cuda else self.model
        self._print_initial_args()
        self._initial_optimizer_schedule()

    def _print_initial_args(self):
        print('-'*18 +  ' initial args ' + '-'*18)
        for k, v in sorted(vars(self.args).items()):
            print(f'--{k} {v}')
        print('-'*50)

    def _initial_optimizer_schedule(self):
        if self.datamodule.dataset(flag = 'train') is None:
            self.datamodule.load_dataset(flag='train')

        # Prepare optimizer
        params = dict(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        paras_new = []
        for k,v in params.items():
            if 'bert' in k:
                if not any(nd in k for nd in no_decay):
                    paras_new += [{'params':[v], 'lr': self.args.bert_learning_rate, 'weight_decay': 0.01}]
                if any(nd in k for nd in no_decay):
                    paras_new += [{'params':[v], 'lr': self.args.bert_learning_rate, 'weight_decay': 0.0}]
            else:
                paras_new += [{'params': [v], 'lr': self.args.learning_rate, 'weight_decay': 0.01}]

        self.optimizer = AdamW(paras_new,
                               correct_bias=False)

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                        num_warmup_steps=self.args.num_warmup_steps,
                                                        num_training_steps=self.datamodule.num_train_steps)

    def get_train_loader(self):
        if self.datamodule.dataset(flag = 'train') is None:
            self.datamodule.load_dataset(flag='train')
        return self.datamodule.get_dataloader(dataset=self.datamodule.dataset(flag='train'))

    def get_valid_loader(self):
        if self.datamodule.dataset(flag = 'valid') is None:
            self.datamodule.load_dataset(flag='valid')
        return self.datamodule.get_dataloader(dataset=self.datamodule.dataset(flag='valid'))

    def get_test_loader(self):
        if self.datamodule.dataset(flag = 'test') is None:
            self.datamodule.load_dataset(flag='test')
        return self.datamodule.get_dataloader(dataset=self.datamodule.dataset(flag='test'))

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        train_loader = self.get_train_loader()

        self.logger.info("-"*20 +  " Epoch: " + str(epoch) + "-"*20)
        self.logger.info("  Num examples = %d", len(train_loader.dataset))
        self.logger.info("  Batch size = %d", self.args.batch_size)

        train_loss, correct_predictions, sample_nums = 0, 0, 0

        for step, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            for key in batch.keys():
                if batch[key] is not None:
                    batch[key] = batch[key].to(self.args.device) if self.is_cuda else batch[key]
            batch_loss, batch_logits = self.model(input_ids=batch['input_ids'],
                                                  input_mask = batch['input_mask'],
                                                  segment_ids = batch['segment_ids'],
                                                  s2_input_ids=batch['s2_input_ids'],
                                                  s2_input_mask=batch['s2_input_mask'],
                                                  s2_segment_ids=batch['s2_segment_ids'],
                                                  img_region_feat = batch['img_region_feat'],
                                                  labels = batch['label_ids']
                                                  )
            _, preds = torch.max(batch_logits, dim=-1)
            correct_predictions += torch.sum(preds == batch['label_ids']).item()

            if self.args.gradient_accumulation_steps > 1:
                batch_loss = batch_loss / self.args.gradient_accumulation_steps
            batch_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            train_loss += batch_loss.item()

            sample_nums += len(batch['input_ids'])

            if step % self.args.gradient_accumulation_steps == 0:
                self.logger.info(f'  step{step} train loss {train_loss/(step+1):.4f} acc {correct_predictions/sample_nums:.4f} '
                                 f'lr {self.optimizer.param_groups[-1]["lr"]:.10f}')
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        train_acc = correct_predictions / len(train_loader.dataset)
        train_loss = train_loss / len(train_loader)

        return train_acc, train_loss


    def do_evaluate(self, test_flag = False):
        self.model.eval()

        if test_flag:
            data_loader = self.get_test_loader()
        else:
            data_loader = self.get_valid_loader()
        eval_loss, eval_acc = 0, 0
        eval_labels, eval_preds = [], []

        with torch.no_grad():
            for batch in tqdm(data_loader):
                for key in batch.keys():
                    if batch[key] is not None:
                        batch[key] = batch[key].to(self.args.device) if self.is_cuda else \
                            batch[key]

                batch_loss, batch_logits = self.model(input_ids=batch['input_ids'],
                                                      input_mask=batch['input_mask'],
                                                      segment_ids=batch['segment_ids'],
                                                      s2_input_ids=batch['s2_input_ids'],
                                                      s2_input_mask=batch['s2_input_mask'],
                                                      s2_segment_ids=batch['s2_segment_ids'],
                                                      img_region_feat=batch['img_region_feat'],
                                                      labels=batch['label_ids']
                                                      )
                _, preds = torch.max(batch_logits, dim=-1)
                eval_acc += torch.sum(preds == batch['label_ids']).item()
                eval_loss += batch_loss.item()

                batch_logits = batch_logits.detach().cpu().numpy()
                label_ids = batch['label_ids'].to('cpu').numpy()
                eval_labels.append(label_ids)
                eval_preds.append(batch_logits)

        eval_loss = eval_loss / len(data_loader)
        eval_acc = eval_acc / len(data_loader.dataset)

        true_label = np.concatenate(eval_labels)
        pred_outputs = np.concatenate(eval_preds)
        precision, recall, F_score = macro_f1(true_label, pred_outputs)
        result = {'loss': eval_loss,
                  'acc': eval_acc,
                  'f1': F_score}

        # if not test_flag:
        #     self.logger.info("***** Dev Eval results *****")
        #     for key in sorted(result.keys()):
        #         self.logger.info("  %s = %s", key, str(result[key]))

        return result



    def do_train(self):
        assert self.optimizer is not None, 'AssertError: Optimizer should be initialed...'
        patience = self.args.patience
        best_valid_f1, best_valid_acc = 0, 0
        best_test_results = {}
        self.logger.info('Training begins....')


        for epoch in range(self.args.num_epoch):
            start = time.time()
            train_acc, train_loss = self.train_epoch(epoch)
            val_res = self.do_evaluate()
            test_res = self.do_evaluate(test_flag=True)
            duration = time.time() - start

            self.logger.info("Final  Train loss {:5.4f} | Train Acc {:5.4f}".format(train_loss, train_acc))
            self.logger.info("-"*19 + " Eval results " + "-"*19)
            self.logger.info(f"Dev acc {val_res['acc']:.4f} loss {val_res['loss']:.4f} f1 {val_res['f1']:.4f} | "
                             f"Test acc {test_res['acc']:.4f} loss {test_res['loss']:.4f} f1 {test_res['f1']:.4f}")
            self.logger.info('-'*50)

            #save model for every epoch
            model_name = 'model_' + str(epoch + 1)
            save_model(self.model, self.args.ckpt_out_file, model_name=model_name)

            if val_res['acc'] >= best_valid_acc:
                best_test_results['acc_acc'] = test_res['acc']
                best_test_results['acc_f1'] = test_res['f1']
                best_valid_acc = val_res['acc']

            if val_res['f1'] >= best_valid_f1:
                best_test_results['f1_acc'] = test_res['acc']
                best_test_results['f1_f1'] = test_res['f1']
                # Save a trained model
                self.logger.info("***** Saving the best ckpt *****")
                patience = self.patience
                save_model(self.model, self.args.ckpt_out_file, model_name='model_b')
                best_valid_f1 = val_res['f1']
            else:
                patience -= 1
            self.logger.info(f'Use {(duration/60):5.4f}mins to run one epoch\n')

            if patience <= 0:
                # print(f"Don't have the patience, break!!!")
                print(f"We are on a break!!!")
                break

        #print results
        print(f'Based on dev acc: test acc {best_test_results["acc_acc"]:.4f} f1 {best_test_results["acc_f1"]:.4f} \n'
              f'Based on dev f1 : test acc {best_test_results["f1_acc"]:.4f} f1 {best_test_results["f1_f1"]:.4f}')

        average_checkpoint.main()

    def do_test(self):
        self.model = load_model(self.args,)
        self.model = self.model.to(self.args.device) if self.is_cuda else self.model
        test_results = self.do_evaluate(test_flag=True)
        print('Final acc {:5.4f} | Final Macro F1 {:5.4f}'.format(test_results['acc'],
                                                                  test_results['f1']))


    def do_test_with_avg_model(self):
        average_checkpoint.main()
        self.model.load_state_dict(load_avg_model_state(self.args))
        self.model = self.model.to(self.args.device) if self.is_cuda else self.model
        test_results = self.do_evaluate(test_flag=True)
        print('Final acc {:5.4f} | Final Macro F1 {:5.4f}'.format(test_results['acc'],
                                                                  test_results['f1']))

