from __future__ import absolute_import
from __future__ import division
import logging
logger = logging.getLogger()
import numpy as np
import torch
from codes.data.feed_data import RelationEntityBatcher
from codes.data.grapher import GrapherBatcher


class Episode():
    """
    run one episode i.e. inference of num_batch query triples on the single complete graph
    graph: Grapher: the single complete graph with unique entities and unique triples (dict[e1] = set((r,e2))) NOT UNIQUE for(e1, r)!
    data: e1(numpy array), r(numpy array), e2(numpy array), allans(list[set of correct answers]) in batches
    return state (et, rl, vd)
    """
    def __init__(self, graph, data, params):
        self.grapher = graph
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode, batcher, hub_node, answer_set = params
        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = num_rollouts
        else:
            self.num_rollouts = test_rollouts
        self.current_hop = 0

        self.positive_reward = torch.tensor(positive_reward, dtype=torch.int64)
        self.negative_reward = torch.tensor(negative_reward, dtype=torch.int64)
        self.use_cuda = torch.cuda.is_available()

        self.batch_question_nodes, self.question_seq, answer, cls_gt, question_str, graph_idx = data

        self.no_examples = len(self.question_seq)
        if self.use_cuda:
            self.positive_reward = self.positive_reward.cuda()
            self.negative_reward = self.negative_reward.cuda()

        self.question_sentences = np.repeat(np.array(question_str, dtype=str), self.num_rollouts, axis=0)

        b = self.grapher.get_graph_data(graph_idx, self.num_rollouts, hub_node, self.mode)
        b.to(torch.device('cuda') if self.use_cuda else torch.device('cpu'))
        self.start_entities = b.start_entities.clone()
        self.b = b
        self._n_soft_labels = self.b.labels.shape[-1]
        self._b_labels = self.b.labels.view(-1, self._n_soft_labels) # ID -> node labels
        self._b_weights = self.b.weights.view(-1, self._n_soft_labels)
        self.final_eos_node = self.b.final_padded

        self.range_arr = torch.arange(len(b.start_entities), dtype=torch.int64)
        if self.mode!='train':
            self.num_rollouts = len(self.start_entities) // self.batch_question_nodes.shape[0]

        self.end_entities = answer.repeat_interleave(self.num_rollouts, 0)  # [B*Rollouts]
        self.cls_gt = cls_gt.repeat_interleave(self.num_rollouts, 0)

        self.global_answer_set = answer_set

        if self.use_cuda:
            self.range_arr = self.range_arr.cuda()
            self.end_entities = self.end_entities.cuda()
            self.global_answer_set = self.global_answer_set.cuda()
            self.cls_gt = self.cls_gt.cuda()

        self.question_rowlength = torch.sum(self.question_seq, dim=1).type(torch.float32)

        next_actions = self.grapher.return_next_actions(b.start_entities)
        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.start_entities
        self.trajectory_print = []
        self.trajectory = []
        self.call_next_step_is_not_allowed=False
        self.last_step_is_label = False

    def get_state(self):
        return self.state

    def get_query_string(self):
        return self.question_sentences

    def state_to_numpy(self, is_binary):
        self.call_next_step_is_not_allowed=True
        if self.last_step_is_label:
            softlabels = None
            globallabels = None
        else:
            softlabels = self.b.labels.view(-1, 3)[self.state['current_entities']].cpu().data.numpy()
            globallabels = self.global_answer_set.cpu().data.numpy()
        if is_binary:
            arg1 = self.b.labels.view(-1, 3)[:,0][self.state['current_entities']].cpu().data.numpy()
        else:
            arg1 = self.state['current_entities'].cpu().data.numpy() # soft-binding pure agent, because last step is label
        return [
            arg1,
            self.b.labels.view(-1, 3)[:,0][self.start_entities].cpu().data.numpy(), # for both
            softlabels, # with classifier, because last step output position/idx not label => labels[pred_idx]
            globallabels,
        ]

    def get_query_relation_compact(self):
        if self.use_cuda:
            self.batch_question_nodes = self.batch_question_nodes.cuda()
            self.question_rowlength = self.question_rowlength.cuda()
            self.question_seq = self.question_seq.cuda()
            self.final_eos_node = self.final_eos_node.cuda()

        self.b.add(question_tokens = self.batch_question_nodes,
              question_rowlength = self.question_rowlength,
              question_seq=self.question_seq)
        return self.b

    def reset(self):
        self.current_entities = self.start_entities
        next_actions = self.grapher.return_next_actions(self.current_entities)

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        return self.state

    def reset_final(self):
        current_entities = self.final_eos_node
        next_actions = self.grapher.return_next_actions(current_entities)
        self.current_entities = self.final_eos_node
        self.state['current_entities'] = self.current_entities
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]

        return self.state

    def reset_chooser(self):
        self.last_step_is_label = True
        current_entities = self.state["current_entities"]
        next_actions = self.grapher.return_entity_chooser_actions(current_entities)
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        return self.state

    def get_local_next_action(self):
        next_action = self._b_labels[self.current_entities]
        next_action_weight = self._b_weights[self.current_entities]
        match = next_action.flatten() == self.end_entities.repeat_interleave(self._n_soft_labels)
        match = match.view(-1, self._n_soft_labels)
        # mask
        mask = match.any(-1)

        # label
        nonzero = torch.nonzero(match)
        target = torch.zeros_like(self.end_entities)
        target[nonzero[:, 0]] = nonzero[:, 1]

        return next_action.clone().detach(), next_action_weight.clone().detach(), target, mask

    def get_global_next_action(self):
        match = (self.end_entities[:, None] == self.global_answer_set[None])
        mask = match.any(1)
        nonzero = torch.nonzero(match)
        target = torch.zeros_like(self.end_entities)
        target[nonzero[:, 0]] = nonzero[:, 1]

        return self.global_answer_set, None, target, mask

    def get_reward(self):
        current_entities = self.current_entities
        reward = (current_entities == self.end_entities)
        # set the True and False values to the values of positive and negative rewards.
        reward = torch.where(reward, self.positive_reward, self.negative_reward) # [B,]
        return reward.type(torch.float32)

    def get_reward_idx(self):
        current_entities = self.b.labels.view(-1, 3)[:,0][self.current_entities]
        reward = (current_entities == self.end_entities)
        # set the True and False values to the values of positive and negative rewards.
        reward = torch.where(reward, self.positive_reward, self.negative_reward) # [B,]
        return reward.type(torch.float32)

    def get_reward_program(self):
        reward = self.get_reward_idx()
        return reward

    def update_beam_trajectory(self, select_idx):
        for j in range(len(self.trajectory)):
            self.trajectory[j] = self.trajectory[j][select_idx]

        for j in range(len(self.trajectory_print)):
            self.trajectory_print[j] = self.trajectory_print[j][select_idx]

    def get_print_trajectory(self):
        l = len(self.trajectory_print)
        relation = [self.trajectory_print[i].cpu().data.numpy() for i in range(0,l,2)]
        entity = [self.trajectory_print[i+1].cpu().data.numpy() for i in range(0,l,2)]
        return entity, relation


    def __call__(self, action, laststep=False):
        assert not self.call_next_step_is_not_allowed
        self.current_hop += 1
        self.current_entities = self.state['next_entities'][self.range_arr, action]
        chosen_relation = self.state['next_relations'][self.range_arr, action]

        if laststep:
            self.state['current_entities'] = self.current_entities
            chosen_relation_embeddings = self.b.edge_labels.view(-1, 3)[:, 0][chosen_relation]
            self.trajectory_print.extend([chosen_relation_embeddings, self.current_entities])
            return self.state

        next_actions = self.grapher.return_next_actions(self.current_entities)

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities

        current_entities_embeddings = self.b.labels.view(-1, 3)[:,0][self.current_entities]
        chosen_relation_embeddings = self.b.edge_labels.view(-1, 3)[:,0][chosen_relation]

        self.trajectory_print.extend([chosen_relation_embeddings, current_entities_embeddings])
        self.trajectory.extend([chosen_relation, self.current_entities])

        return self.state

    def __call__b(self, action, laststep=False, isbinary=True):
        self.current_hop += 1
        chosen_relation = self.state['next_relations'][self.range_arr, action]

        self.current_entities = self.state['next_entities'][self.range_arr, action]
        self.current_entities_embeddings = self.state['next_entities_embeddings'][self.range_arr, action]

        self.state['current_entities'] = self.current_entities
        self.state['current_entities_embeddings'] = self.current_entities_embeddings

        if not laststep:
            next_actions = self.grapher.return_next_actions(self.current_entities)
            self.state['next_relations'] = next_actions[:, :, 1]
            self.state['next_entities'] = next_actions[:, :, 0]
            self.state['next_entities_embeddings'] = next_actions[:, :, 2]

        self.trajectory.extend([chosen_relation, self.current_entities])
        self.trajectory_print.extend([chosen_relation, self.current_entities_embeddings])

        return self.state



class env():
    """
    initialize knowledge graph, self.get_episode() return episode object which runs one episode of batch queries triples
    on the single complete graph
    """
    def __init__(self, params, mode='train'):
        """
        num_rollouts: int
        positive reward: int 1
        negative reward: int 0
        mode: train test dev
        path_len: int 2 maximum reasoning length
        """
        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.mode = mode
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        input_dir = params['data_input_dir']
        self.hub_node = not params['random_start']
        self.streaming = params['streaming']
        self.answer_set = torch.tensor([params['embedding_vocab'].get(x, 0) for x in params['answer_set']], dtype=torch.long)
        self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                             batch_size=params['batch_size'],
                                             embedding_vocab = params['embedding_vocab'],
                                             relation_vocab=params['relation_vocab'],
                                             mode=self.mode,
                                             is_binary= params['case0']
                                             )
        self.batches_per_epoch = self.batcher.batches_per_epoch
        self.total_no_examples = len(self.batcher.store)

        self.grapher = GrapherBatcher(input_dir = params['data_input_dir'],
                                      mode = mode,
                                      embedding_vocab=params['embedding_vocab'],
                                      relation_vocab=params['relation_vocab'],
                                      max_num_actions=params['max_num_actions'],
                                      hub_node=self.hub_node,
                                      streaming=self.streaming
                                      )

    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, \
                 self.negative_reward, self.mode, self.batcher, self.hub_node, self.answer_set.clone()
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():

                yield Episode(self.grapher, data, params)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                yield Episode(self.grapher, data, params)
