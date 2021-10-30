import json
import numpy as np
import csv
import os
import torch
import logging
logger = logging.getLogger()
from collections import OrderedDict


class ProcessData():
    def __init__(self):
        self.replace_list = OrderedDict({
            "?": " ?",
            ",": " ,",
            "_": " ",
            "n't": " n't", # aren't, isn't
            "'s": " 's", # what's, bird's
            "s'": "s '", # birds'
            "tshirt": "t-shirt"
        })

    def _canonicalization(self, sentence):
        origin = sentence
        sentence = sentence.lower()
        for token in self.replace_list:
            sentence = sentence.replace(token, self.replace_list[token])

        debug = False
        if debug:
            old = origin
            new = sentence
            for t in [' ?', '?', ' ,', ',', " 's", "'s", "isn't", "is not"]:
                old = old.replace(t, '')
                new = new.replace(t, '')
            if old.lower() != new:
                print(sentence, origin)
        return sentence


class RelationEntityBatcher():
    """
    loading and batching questions
    inputs:
        input_dir: path to data root folder(which contains splitted data folder - train, validation, test)
        unique_name_vocab: [dict] loaded by json.load(open('unique_name_vocab.json')) word to embedding index
    outputs:
        list of encoded questions and their answers (both are represented by embedding id)
    """
    def __init__(self, input_dir, batch_size, embedding_vocab, relation_vocab,
                 mode ="train", is_binary=False):
        self.input_path = os.path.join(input_dir, mode+'.txt')
        graph_list = json.load(open(input_dir + '/split_index.json'))
        self.graph_list = [k for k,v in graph_list.items() if v==mode]
        self.graph_list = set(self.graph_list)
        self.batch_size = batch_size
        self.embedding_vocab = embedding_vocab #:String->IDX
        self.relation_vocab = relation_vocab
        self.mode = mode
        self._ans_types = {'yes':1, 'no':2}
        self._ans_embs = {'yes': embedding_vocab['TRUE'], 'no':embedding_vocab['FALSE']}
        self.is_binary = is_binary
        self.process = ProcessData()
        self.create_word2vec(self.input_path)
        self.get_statistics()
        self.batches_per_epoch = len(self.store)//self.batch_size + 1

    def get_next_batch(self):
        if self.mode == 'train':
            yield self.yield_next_batch_train()
        else:
            yield self.yield_next_batch_test()

    def pad_store_data_torch(self, store):
        """
        inputs: all question token idx: list of list [[11,28], [3,4,9]]
        outputs: torch.tensor
        """
        pad_length = max([len(x) for x in store])
        batch_size = len(store)
        store_padded = torch.zeros(batch_size, pad_length, dtype=torch.int64)
        store_rowlength = torch.zeros(batch_size, pad_length, dtype=torch.int64)
        for i in range(batch_size):
            valid = len(store[i])
            store_padded[i, :valid] = torch.tensor(store[i])
            store_rowlength[i, :valid] = 1
        return store_padded, store_rowlength

    def _binary_question(self, ans):
        return ans == 'yes' or ans == 'no'

    def create_word2vec(self, input_file):
        question_strings = []
        questions = []
        answers = []
        answer_types = []
        graph_ids = []
        logger.info('loading questions from {}'.format(input_file))
        with open(input_file) as raw_input_file:
            csv_file = csv.reader(raw_input_file, delimiter = '\t' )
            for line in csv_file:
                sid, qid, q, ans = line
                if not sid in self.graph_list:
                    continue

                if self.is_binary != self._binary_question(ans):
                    continue

                if self.is_binary:
                    emb = self._ans_embs[ans]
                else:
                    emb = self.embedding_vocab[ans]

                answer_types.append(self._ans_types.get(ans, 0))

                answers.append(emb) # vocabIdx->wordIdx->glove embedding, this is wordIdx

                question_strings.append( sid + '_' + q)

                q = self.process._canonicalization(q)
                tokens = q.split()
                if not all([t in self.embedding_vocab for t in tokens]):
                    print("!!!!!!WARNING word not in vocab")
                tokens = [self.embedding_vocab.get(t, 'PAD') for t in tokens]
                questions.append(tokens)
                graph_ids.append(sid)

        if not all([answers, question_strings, graph_ids, questions]):
            raise RuntimeError('Question file is empty!')

        # convert questions to idx
        self.question_strings = np.array(question_strings)
        self.store = questions
        self.graph_ids = np.array(graph_ids)
        self.store_padded, self.store_rowlength = self.pad_store_data_torch(self.store)
        self.answers = torch.tensor(answers, dtype=torch.int64)

        self.answers_type = torch.tensor(answer_types, dtype=torch.int64)

    def yield_next_batch_train(self):
        start_idx = 0
        while True:
            batch_idx = np.random.randint(0, len(self.store), size=self.batch_size)
            # batch_idx = np.arange(start_idx, start_idx+self.batch_size, 1) % self.store.shape[0]
            start_idx+=self.batch_size
            batch_question_nodes = self.store_padded[batch_idx]
            batch_question_mask = self.store_rowlength[batch_idx]

            batch_y = self.answers[batch_idx]
            batch_y_type = self.answers_type[batch_idx]
            questions = [self.question_strings[x] for x in batch_idx]
            graphs = self.graph_ids[batch_idx]
            yield batch_question_nodes, batch_question_mask, batch_y, batch_y_type, questions,graphs

    def yield_next_batch_test(self):
        remaining_triples = len(self.store)
        current_idx = 0
        while True:
            if remaining_triples == 0:
                return

            if remaining_triples - self.batch_size > 0:
                batch_idx = np.arange(current_idx, current_idx + self.batch_size)
                current_idx += self.batch_size
                remaining_triples -= self.batch_size
            else:
                batch_idx = np.arange(current_idx, len(self.store))
                remaining_triples = 0

            batch_question_nodes = self.store_padded[batch_idx]
            batch_question_mask = self.store_rowlength[batch_idx]

            batch_y = self.answers[batch_idx]
            batch_y_type = self.answers_type[batch_idx]
            questions = [self.question_strings[x] for x in batch_idx]
            graphs = self.graph_ids[batch_idx]

            yield batch_question_nodes, batch_question_mask, batch_y, batch_y_type, questions,graphs

    def get_statistics(self):
        logger.info('total number of questions: {} for {}'.format(len(self.question_strings), self.mode))


class RelationEntityTester():
    def __init__(self, embedder=None):
        self.embedder = embedder # unique_name_vocab:String->IDX

    def return_data(self, question, graph_idx):
        tokens = self._normalize(question)
        return tokens, [1]*len(tokens), [0], [question], [graph_idx]

    def return_data_torch(self, question, graph_idx):
        tokens = self._normalize(question)
        tokens = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0)
        seq = torch.ones_like(tokens)
        ans = torch.tensor([0])
        return tokens, seq, ans, [question], [graph_idx]

    def _normalize(self, q):
        if "'" in q:
            q = q.replace("n't", " not")
            q = q.replace("'s", " is")
        tokens = [x.lower() for x in q.split()]
        if not all([t in self.embedder for t in tokens]):
            print("Warning: word not in vocabulary")
        tokens = [self.embedder.get(t, 0) for t in tokens]
        return tokens
