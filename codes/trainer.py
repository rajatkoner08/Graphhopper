from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import os
import json
import codecs
from collections import defaultdict
from copy import deepcopy
import resource
from tqdm import tqdm
import torch
import numpy as np
from tensorboardX import SummaryWriter

from codes.model.agent import Agent
from codes.model.loss import PolicyGradientLoss
from codes.model.loss import AdvantageEstimator
from codes.data.environment import env
from codes.model.encoder import StaticTokenEncoder
from codes.parse_config import Config
from codes.optim.weight_decay import ExponentialDecay

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

np.set_printoptions(precision=3, suppress=True)


class Trainer():
    def __init__(self, config):
        self.config = config
        params = config.asdict()
        self.params = params
        self.word_embedder=StaticTokenEncoder(wv_dir=params['wv_dir'],
                                               wv_dim=params['embedding_size'],
                                               max_word_length=params['max_sentence_length'])
        self.sorted_entity_labels = params['sorted_entity_labels']
        self.sorted_relations = params['sorted_relations']

        self.use_cuda = torch.cuda.is_available()

        self.case1, self.case2, self.case3 = config.case1, config.case2, config.case3
        self.case0 = config.case0 #binary
        assert sum([self.case1, self.case2, self.case3, self.case0])==1

        self.agent = Agent(config)

        if self.use_cuda:
            self.agent = self.agent.cuda()

        self.save_path = config.model_dir + '/model.pth'
        if not config.skip_training:
            self.train_environment = env(params, 'train')
            self.dev_test_environment = env(params, 'val')
            self.test_environment = self.dev_test_environment
        try:
            self.test_test_environment = env(params, 'test')
        except:
            logger.info('test set is not prepared')
        if config.skip_training:
            self.train_environment = self.test_test_environment
            self.test_environment = self.test_test_environment

        self.random_start = self.config.random_start
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rev_label_vocab = self.train_environment.grapher.rev_embedding_vocab
        self.max_hits_at_1 = 0

        self.advantage_estimator = AdvantageEstimator(config.path_length + int(self.case1+self.case2+self.case0), config.gamma, config.Lambda)
        self.decaying_beta = ExponentialDecay(config.beta, config.decay_rate,
                                              self.train_environment.batches_per_epoch)
        self.policy_gradient_loss = PolicyGradientLoss(self.decaying_beta)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def initialize_weights(self):
        entity_embeddings, relation_embeddings = None, None
        if self.config.pretrained_entity:
            entity_embeddings = self.word_embedder.embedd_entity_batch(self.sorted_entity_labels)
            entity_embeddings[0]*=0

        if self.config.pretrained_relation:
            relation_embeddings = self.word_embedder.embedd_entity_batch(self.sorted_relations)
            relation_embeddings[0] *=0

        self.agent.init_weights(entity_embeddings, relation_embeddings)

    def load_weights(self, model_path):
        map_location = 'cuda' if self.use_cuda else 'cpu'
        ckpt = torch.load(model_path, map_location=map_location)
        if 'agent' in ckpt: # new version with classifier
            self.agent.load_state_dict(ckpt['agent'])
        else:
            self.agent.load_state_dict(ckpt)
        logger.info('load pretrained model from {}'.format(model_path))

    def save_weights(self, model_path):
        ckpt = {
            'agent': self.agent.state_dict(),
        }
        torch.save(ckpt, model_path)

        logger.info('model saved at {}'.format(model_path))

    def train(self):
        if torch.cuda.is_available():
            self.agent = self.agent.cuda()
        self.agent.train()
        loss_history = list()
        train_loss = 0.0
        self.batch_counter = 0
        self.writer = SummaryWriter(logdir=self.config.output_dir)
        for episode in self.train_environment.get_episodes():
            self.batch_counter += 1
            # inputs
            batchdata = episode.get_query_relation_compact()
            state = episode.get_state()
            self.optimizer.zero_grad()
            # 1. init
            self.agent.init(batchdata, state['current_entities'].shape[0])
            # 2. rnn
            for i in range(self.config.path_length):
                if i<self.config.path_length and i % self.config.reset_after == 0:
                    state = episode.reset()

                next_relations = state['next_relations']
                next_entities = state['next_entities']
                current_entities = state['current_entities']
                probs, logits, chosen_relation, idx, out = self.agent(
                    next_relations, next_entities, current_entities, debug=False, laststep=False)
                state = episode(idx, False)

            # 3. final
            # 3.1 agent
            if self.case1: # agent || simple reward
                state = episode.reset_chooser()
                next_relations, next_entities,current_entities = state['next_relations'],state['next_entities'],state['current_entities']
                probs, logits, chosen_relation, idx, out = self.agent(next_relations, next_entities, current_entities, laststep=True)
                state = episode(idx, True)
                rewards = episode.get_reward()

            elif self.case0: #binary
                state = episode.reset_final()
                next_relations, next_entities,current_entities = state['next_relations'],state['next_entities'],state['current_entities']
                probs, logits, chosen_relation, idx, out = self.agent(next_relations, next_entities, current_entities, laststep=False)
                state = episode(idx, False)
                rewards = episode.get_reward_program()

            else:
                raise RuntimeError

            normalized_rewards = self.advantage_estimator.step(rewards)
            normalized_rewards = normalized_rewards.clone().detach()
            loss, reg_loss = self.policy_gradient_loss(probs, logits, normalized_rewards, self.batch_counter)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()

            nll_loss_local, nll_loss_global, global_acc, local_acc = -1, -1, -1, -1

            # print metrics
            train_loss = 0.98 * train_loss + 0.02 * loss.item()

            rewards = rewards.data.cpu().numpy()
            reg_loss = reg_loss.data.cpu().numpy()
            avg_reward = np.mean(rewards)

            reward_reshape = np.reshape(rewards, (self.config.batch_size, self.config.num_rollouts))  # [orig_batch, num_rollouts]
            reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
            reward_reshape = (reward_reshape > 0)
            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")
            # still run on every iteration to avoid memory leak
            loss_history.append([avg_reward, train_loss, nll_loss_local, nll_loss_global, local_acc, global_acc])

            if self.batch_counter % 100 == 0:
                logger.info("batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                            "cls local {3:7.4f} {6:7.4f}, cls global {4:7.4f} {7:7.4f}, agent loss {5:7.4f}".
                            format(self.batch_counter, np.sum(rewards), avg_reward, nll_loss_local,
                                  nll_loss_global,train_loss,local_acc, global_acc)

                            )
                avg_reward_acc, train_loss_acc, cls_local_loss, cls_global_loss, cls_local_acc, cls_global_acc = np.mean(np.array(loss_history), axis=0)
                self.writer.add_scalar("train/accuracy", avg_reward_acc, self.batch_counter)
                self.writer.add_scalar("train/gradient", train_loss_acc, self.batch_counter)
                self.writer.add_scalar("train/entropy", reg_loss, self.batch_counter)
                self.writer.add_scalar("cls/cls_local", cls_local_loss, self.batch_counter)
                self.writer.add_scalar("cls/cls_global", cls_global_loss, self.batch_counter)
                self.writer.add_scalar("cls/acc_local", cls_local_acc, self.batch_counter)
                self.writer.add_scalar("cls/acc_global", cls_global_acc, self.batch_counter)

                self.writer.flush()
                loss_history.clear()
                if self.config.logweight:
                    for i, parameter in enumerate(self.agent.parameters()):
                        self.writer.add_histogram('histogram/' + str(i), parameter.clone(), self.batch_counter)

                logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            if self.batch_counter % self.config.eval_every == 0:
                with open(self.config.output_dir + '/scores.txt', 'a') as score_file:
                    score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                os.mkdir(self.config.path_logger_file + "/" + str(self.batch_counter))
                self.path_logger_file_ = self.config.path_logger_file + "/" + str(self.batch_counter) + "/paths"
                with torch.no_grad():
                    self.test(debug=False, print_paths=False, batch_number=self.batch_counter)
                    self.agent.train()


            if self.batch_counter >= self.config.total_iterations:
                self.writer.add_hparams(
                    {k:str(self.params[k]) for k in self.config.hparam_log},
                    {'hits1': self.max_hits_at_1}
                )
                self.writer.close()
                break

    def test(self, debug=False, print_paths=False, save_model = True,batch_number=None):
        """
        main test function
        """
        self.agent.eval()

        batch_counter = 0
        paths = defaultdict(list)
        answers = []
        predict = defaultdict(list)
        feed_dict = {}
        all_final_reward_1 = 0
        all_final_reward_3 = 0
        all_final_reward_5 = 0
        all_final_reward_10 = 0
        all_final_reward_20 = 0
        auc = 0

        q_pred = []
        total_examples = self.test_environment.total_no_examples
        for episode in tqdm(self.test_environment.get_episodes()):
            batch_counter += 1
            temp_batch_size = episode.no_examples

            # inputs
            batchdata = episode.get_query_relation_compact()
            self.query_string = episode.get_query_string()

            state = episode.get_state()
            total_ans = state['current_entities'].shape[0]
            ans_per_batch = total_ans//temp_batch_size

            if self.random_start:
                self.valid_ans_per_batch = [x*self.config.test_rollouts for x in batchdata.pad_keeper]
            else:
                self.valid_ans_per_batch = [ans_per_batch for _ in range(temp_batch_size)]

            ####logger code####
            if print_paths:
                self.entity_trajectory = []
                self.relation_trajectory = []
                self.entity_trajectory_new = []
                self.relation_trajectory_new = []
            ####################
            self.log_probs = np.ones((total_ans,), dtype=np.float32)
            beam_probs = np.zeros((total_ans, 1))

            ##################### start of inference ####################
            self.agent.init(batchdata, num_paths = state['current_entities'].shape[0], eval=True)
            for i in range(self.config.path_length):
                if i < self.config.path_length and i % self.config.reset_after == 0:
                    state = episode.reset()

                next_relations = state['next_relations']
                next_entities = state['next_entities']
                current_entities = state['current_entities']
                probs, logits, chosen_relation, idx, out = self.agent(
                    next_relations, next_entities, current_entities, debug=debug, laststep=False)

                #beam
                if self.config.beam:
                    k = ans_per_batch
                    new_scores = logits[-1].cpu().data.numpy() + beam_probs #[b*r, m]
                    if i == 0:
                        idx = np.argsort(new_scores)
                        idx = idx[:, -k:] #[b*r, r]
                        ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
                        idx = idx[np.arange(k * temp_batch_size), ranged_idx] #[b*r,1]
                    else:
                        idx = self.top_k(new_scores, k) #[b*r,1]

                    y = idx // self.config.max_num_actions
                    x = idx % self.config.max_num_actions

                    y += np.repeat([b * k for b in range(temp_batch_size)], k)
                    state['current_entities'] = state['current_entities'][y]
                    state['next_relations'] = state['next_relations'][y, :]
                    state['next_entities'] = state['next_entities'][y, :]
                    self.agent.prev_state = (self.agent.prev_state[0][:, y, :], self.agent.prev_state[1][:,y,:])
                    idx = x
                    chosen_relation = state['next_relations'][np.arange(temp_batch_size * k), x]
                    beam_probs = new_scores[y, x]
                    beam_probs = beam_probs.reshape((-1, 1))
                    #beam##
                    self.agent.prev_relation = chosen_relation
                    episode.update_beam_trajectory(y)

                ####logger code####
                if print_paths:
                    self.entity_trajectory_new.append(out[0])
                    self.relation_trajectory_new.append(out[1])
                ####################
                state = episode(idx, False)
                self.log_probs += logits[-1][np.arange(self.log_probs.shape[0]), idx].cpu().data.numpy()

            if self.case1: # agent || simple reward
                state = episode.reset_chooser()
                next_relations, next_entities,current_entities = state['next_relations'],state['next_entities'],state['current_entities']
                probs, logits, chosen_relation, idx, out = self.agent(next_relations, next_entities, current_entities, laststep=True)
                state = episode(idx, True)

            elif self.case0: #binary
                state = episode.reset_final()
                next_relations, next_entities, current_entities = state['next_relations'], state['next_entities'], state['current_entities']
                probs, logits, chosen_relation, idx, out = self.agent(next_relations, next_entities, current_entities, laststep=False)
                state = episode(idx, False)

            #################### end of inference ####################

            if self.config.beam:
                self.log_probs = beam_probs

            ####Logger code####
            if print_paths:
                self.entity_trajectory, self.relation_trajectory = episode.get_print_trajectory()
            ####################

            ce, se, ce_soft, ce_soft_global = episode.state_to_numpy(self.case0)
            ce = ce.reshape((temp_batch_size, -1))
            se = se.reshape((temp_batch_size, -1))

            #################### ask environment for final reward ####################
            if self.case1:
                rewards = episode.get_reward()  # [B*test_nodes_rollouts]
                rewards = rewards.cpu().data.numpy()
                reward_reshape = rewards.reshape(temp_batch_size, ans_per_batch)  # [orig_batch, test_nodes_rollouts]

            elif self.case0:
                rewards = episode.get_reward_idx()

                reward_reshape = torch.reshape(rewards, (temp_batch_size, ans_per_batch))  # [orig_batch, test_nodes_rollouts]
                rewards = rewards.cpu().data.numpy()
                reward_reshape = reward_reshape.cpu().data.numpy()
            ####################

            assert ce.shape==(temp_batch_size,ans_per_batch)
            assert reward_reshape.shape==(temp_batch_size,ans_per_batch), reward_reshape.shape

            self.log_probs = np.reshape(self.log_probs, (temp_batch_size, -1))

            for b in range(temp_batch_size):
                self.log_probs[b, self.valid_ans_per_batch[b]:] = -1e9
            sorted_indx = np.argsort(-self.log_probs)


            final_reward_1 = 0
            final_reward_3 = 0
            final_reward_5 = 0
            final_reward_10 = 0
            final_reward_20 = 0
            AP = 0

            for b in range(temp_batch_size):
                answer_pos = None
                seen = set()
                pos = 0
                if self.config.pool == 'max':
                    sorted_answers = []
                    for r in sorted_indx[b]:
                        sorted_answers.append(ce[b, r])
                        if reward_reshape[b, r] == self.config.positive_reward:
                            answer_pos = pos
                            break
                        if ce[b, r] not in seen:
                            seen.add(ce[b, r])
                            pos += 1

                if answer_pos != None:
                    if answer_pos < 20:
                        final_reward_20 += 1
                        if answer_pos < 10:
                            final_reward_10 += 1
                            if answer_pos < 5:
                                final_reward_5 += 1
                                if answer_pos < 3:
                                    final_reward_3 += 1
                                    if answer_pos < 1:
                                        final_reward_1 += 1

                if answer_pos == None:
                    AP += 0
                else:
                    AP += 1.0 / ((answer_pos + 1))
                if print_paths:
                    start_e = self.rev_label_vocab[se[b,0]]
                    end_e = self.rev_label_vocab[episode.end_entities[b * ans_per_batch].item()]

                    qr = self.query_string[b * ans_per_batch]
                    paths[str(qr)].append(str(start_e) + "\t" + str(end_e) + "\n")
                    paths[str(qr)].append("Reward:" + str(1 if answer_pos != None and answer_pos < 1 else 0) + "\n")
                    for r in sorted_indx[b]:
                        if r >= self.valid_ans_per_batch[b]:
                            break
                        indx = b * ans_per_batch + r
                        if reward_reshape[b,r] == self.config.positive_reward:
                            rev = 1
                        else:
                            rev = -1
                        answers.append(
                            self.rev_label_vocab[se[b, r]] + '\t' + self.rev_label_vocab[ce[b, r]] + '\t' + str(
                                self.log_probs[b, r]) + '\n')

                        estr = '\t'.join([str(self.rev_label_vocab[e[indx]]) for e in self.entity_trajectory])
                        rstr = '\t'.join([str(self.rev_relation_vocab[re[indx]]) for re in self.relation_trajectory])

                        paths[str(qr)].append( estr
                             + '\n' + rstr + '\n' + str(
                                rev) + '\n' + str(
                                self.log_probs[b, r]) + '\n___' + '\n')

                    paths[str(qr)].append("#####################\n")
                    if answer_pos is None or answer_pos >= 1:
                        predict['wrong'].append(
                            str(qr) + '\t' + self.rev_label_vocab.get(sorted_answers[0], 'NOTI') + '\n')
                    else:
                        predict['correct'].append(
                            str(qr) + '\t' + self.rev_label_vocab.get(sorted_answers[0], 'NOTI') + '\n')

            all_final_reward_1 += final_reward_1
            all_final_reward_3 += final_reward_3
            all_final_reward_5 += final_reward_5
            all_final_reward_10 += final_reward_10
            all_final_reward_20 += final_reward_20
            auc += AP

        all_final_reward_1 /= total_examples
        all_final_reward_3 /= total_examples
        all_final_reward_5 /= total_examples
        all_final_reward_10 /= total_examples
        all_final_reward_20 /= total_examples
        auc /= total_examples
        if save_model:
            if all_final_reward_1 >= self.max_hits_at_1:
                self.max_hits_at_1 = all_final_reward_1
                self.save_weights(self.save_path)

        if print_paths:
            logger.info("[ printing paths at {} ]".format(self.config.output_dir + '/test_beam/'))
            for q in paths:
                j = q.replace('/', '-')
                with codecs.open(self.path_logger_file_ + '_' + j, 'a', 'utf-8') as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            with open(self.path_logger_file_ + 'answers', 'w') as answer_file:
                for a in answers:
                    answer_file.write(a)
            with open(self.path_logger_file_ + 'wrong', 'w') as predict_file:
                for q in predict['wrong']:
                    predict_file.write(q)
            with open(self.path_logger_file_ + 'correct', 'w') as predict_file:
                for q in predict['correct']:
                    predict_file.write(q)
            with open(self.path_logger_file_ + 'binaryquery', 'w') as predict_file:
                for q in q_pred:
                    predict_file.write(q+'\n')


        with open(self.config.output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1))
            score_file.write("\n")
            score_file.write("Hits@3: {0:7.4f}".format(all_final_reward_3))
            score_file.write("\n")
            score_file.write("Hits@5: {0:7.4f}".format(all_final_reward_5))
            score_file.write("\n")
            score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10))
            score_file.write("\n")
            score_file.write("Hits@20: {0:7.4f}".format(all_final_reward_20))
            score_file.write("\n")
            score_file.write("auc: {0:7.4f}\n".format(auc))
            score_file.write("\n")
            score_file.write("\n")

        logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
        logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
        logger.info("Hits@5: {0:7.4f}".format(all_final_reward_5))
        logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
        logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
        logger.info("auc: {0:7.4f}".format(auc))
        if batch_number:
            self.writer.add_scalar("valid/accuracy", all_final_reward_1, self.batch_counter)
            self.writer.flush()
        return all_final_reward_1

    def top_k(self, scores, k):
        scores = scores.reshape(-1, k * self.config.max_num_actions)  # [B, (k*max_num_actions)]
        idx = np.argsort(scores, axis=1)
        idx = idx[:, -k:]  # take the last k highest indices # [B , k]
        return idx.reshape((-1))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="", type=str)
    parsed = parser.parse_args()

    config = Config(parsed.config)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(config.log_file_name, 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info('Total number of labels {}'.format(len(set(config.embedding_vocab.values()))))
    logger.info('Total number of relations {}'.format(len(config.relation_vocab)))
    save_path = ''

    hyperparams = deepcopy(config.asdict())
    json.dump(hyperparams, open(config.output_dir+'/hyperparams.json', 'w'))

    trainer = Trainer(config)

    #Training
    if not config.skip_training:
        if config.load_model:
            trainer.load_weights_v1(config.model_load_dir)
        else:
            trainer.initialize_weights()
        trainer.train()
        model_path = trainer.save_path
    else:
        model_path = config.model_load_dir

    # Test (Only if test set is prepared)
    logger.info("Test: Loading model from {}".format(model_path))
    trainer.load_weights(model_path)

    output_dir = config.output_dir
    os.mkdir(config.path_logger_file + "/" + "test_beam")
    trainer.path_logger_file_ = config.output_dir + "/" + "test_beam" + "/paths"
    with open(output_dir + '/scores.txt', 'a') as score_file:
        score_file.write("Test (beam) scores with best model from " + save_path + "\n")

    trainer.test_environment = trainer.test_test_environment
    hits1 = trainer.test(debug=False, print_paths=True, save_model=False)
    hyperparams['hits1'] = hits1
    json.dump(hyperparams, open(output_dir+'/hyperparams.json', 'w'))
