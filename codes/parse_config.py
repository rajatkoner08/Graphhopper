from __future__ import absolute_import
from __future__ import division
import uuid
import os
from pprint import pprint
import json
from pathlib import Path
import yaml


class Config():
    def __init__(self, config_file, save_config=True):
        with open(config_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.Loader)

        parsed = {}
        ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent
        parsed["data_input_dir"] = os.path.join(ROOT_DIR, data.get("data_input_dir", ""))
        parsed["input_file"] = data.get("input_file", "train.txt")
        parsed["create_vocab"] = int(data.get("create_vocab", 0))
        parsed["vocab_dir"] = str(data.get("vocab_dir", ""))
        parsed["wv_dir"] = os.path.join(ROOT_DIR, data.get("wv_dir", "wordVector"))
        parsed["model_load_dir"] = os.path.join(ROOT_DIR, data.get("model_load_dir", ""))
        parsed["load_model"] = int(data.get("load_model", 0))
        parsed["max_num_actions"] = int(data.get("max_num_actions", 200))
        parsed["max_num_entities"] = -1
        parsed["max_sentence_length"] = data.get("max_sentence_length", 30)
        parsed["logweight"] = data.get("logweight", False)
        parsed["skip_training"] = data.get("skip_training", False)
        parsed["path_length"] = int(data.get("path_length", 3))
        parsed["reset_after"] = int(data.get("reset_after", 4))
        parsed["hidden_size"] = int(data.get("hidden_size", 50))
        parsed["embedding_size"] = int(data.get("embedding_size", 50))
        parsed["n_head"] = int(data.get("n_head", 8))
        parsed["batch_size"] = int(data.get("batch_size", 128))
        parsed["grad_clip_norm"] = int(data.get("grad_clip_norm", 5))
        parsed["l2_reg_const"] = float(data.get("l2_reg_const", 1e-2))
        parsed["learning_rate"] = float(data.get("learning_rate", 1e-3))
        parsed["beta"] = float(data.get("beta", 1e-2))
        parsed["positive_reward"] = float(data.get("positive_reward", 1.0))
        parsed["negative_reward"] = float(data.get("negative_reward", 0))
        parsed["gamma"] = float(data.get("gamma", 1))
        # parsed["log_dir"] = str(data.get("log_dir", "./logs/"))
        parsed["num_rollouts"] = int(data.get("num_rollouts", 20))
        parsed["test_rollouts"] = int(data.get("test_rollouts", 2))
        parsed["LSTM_layers"] = int(data.get("LSTM_layers", 1))
        parsed["base_output_dir"] = os.path.join(ROOT_DIR, data.get("base_output_dir", ""))
        parsed["total_iterations"] = int(data.get("total_iterations", 2000))
        parsed["Lambda"] = float(data.get("Lambda", 0.0))
        parsed["pool"] = str(data.get("pool", "sum"))
        parsed["eval_every"] = int(data.get("eval_every", 100))
        parsed["use_entity_embeddings"] = int(data.get("use_entity_embeddings", 1))
        parsed["train_entity_embeddings"] = int(data.get("train_entity_embeddings", 0))
        parsed["train_relation_embeddings"] = int(data.get("train_relation_embeddings", 1))
        parsed["nell_evaluation"] = int(data.get("nell_evaluation", 0))
        parsed["random_start"] = int(data.get("random_start", 1))
        parsed["streaming"] = int(data.get("streaming", 0))
        parsed["pretrained_entity"] = int(data.get("pretrained_entity", 0))
        parsed["pretrained_relation"] = int(data.get("pretrained_relation", 0))
        parsed["encoder_type"] = str(data.get("encoder_type", "gcn"))
        parsed["decoder_type"] = str(data.get("decoder_type", "none"))
        parsed["fusion_type"] = list(data.get("fusion_type", ["mean1", "dot1"]))
        parsed["transformer_nlayers"] = int(data.get("transformer_nlayers", 2))
        parsed["transformer_d_k"] = int(data.get("transformer_d_k", 64))
        parsed["dropout"] = float(data.get("dropout", 0.))
        parsed["weight_decay"] = float(data.get("weight_decay", 0.))
        parsed["decay_rate"] = float(data.get("decay_rate", 0.9))
        parsed["beam"] = data.get("beam", 0) == 1
        parsed["agg_mode"] = data.get("agg_mode", "none")
        parsed["classifier"] = data.get("classifier", "case1")
        parsed['hparam_log'] = ['embedding_size', 'beta', 'decay_rate', 'LSTM_layers', 'Lambda', 'gamma', 'path_length',
                                'encoder_type', 'fusion_type', 'transformer_nlayers', 'weight_decay', 'dropout',
                                'load_model', "agg_mode", "classifier"]

        # in case of typo
        assert all([x in parsed for x in data.keys()]), "invalid input args"

        if parsed["beam"] and parsed['max_num_actions']<parsed['test_rollouts']:
            raise RuntimeError(f'If use beam, test rollouts {parsed["test_rollouts"]} should be smaller than  max number of actions {parsed["max_num_actions"]}')

        parsed['input_files'] = [parsed['data_input_dir'] + '/' + parsed['input_file']]

        parsed['use_entity_embeddings'] = (parsed['use_entity_embeddings'] == 1)
        parsed['train_entity_embeddings'] = (parsed['train_entity_embeddings'] == 1)
        parsed['train_relation_embeddings'] = (parsed['train_relation_embeddings'] == 1)

        parsed['random_start'] = (parsed['random_start'] == 1)
        if parsed['random_start'] and parsed['beam']:
            raise RuntimeError('Beam not supported for random start')


        parsed['pretrained_embeddings_action'] = ""
        parsed['pretrained_embeddings_entity'] = ""

        parsed['pretrained_entity'] = (parsed['pretrained_entity'] == 1)
        parsed['pretrained_relation'] = (parsed['pretrained_relation'] == 1)

        parsed['output_dir'] = parsed['base_output_dir'] + '/' + str(uuid.uuid4())[:4]+'_'+str(parsed['path_length'])+'_'+str(parsed['beta'])+'_'+str(parsed['test_rollouts'])+'_'+str(parsed['Lambda'])
        parsed['model_dir'] = parsed['output_dir']+'/'+ 'model/'
        parsed['load_model'] = (parsed['load_model'] == 1)

        parsed["case1"] = parsed["classifier"]=="case1"
        parsed["case2"] = parsed["classifier"]=="case2"
        parsed["case3"] = parsed["classifier"]=="case3"
        parsed["case0"] = parsed["classifier"]=="case0"

        ##Logger##
        if save_config:
            parsed['path_logger_file'] = parsed['output_dir']
            parsed['log_file_name'] = parsed['output_dir'] +'/log.txt'
            os.makedirs(parsed['output_dir'])
            os.mkdir(parsed['model_dir'])
            with open(parsed['output_dir']+'/config.txt', 'w') as out:
                pprint(parsed, stream=out)

            # print and return
            maxLen = max([len(ii) for ii in parsed.keys()])
            fmtString = '\t%' + str(maxLen) + 's : %s'
            print('Arguments:')
            for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

        # load vocabularies
        parsed['relation_vocab'] = json.load(open(parsed['data_input_dir'] + '/relation_vocab.json'))
        parsed['embedding_vocab'] = json.load(open(parsed['data_input_dir'] + '/embedding_vocab.json'))

        parsed['JUDGE'] = parsed['relation_vocab']['JUDGE']
        parsed['TRUE'] = parsed['embedding_vocab']['TRUE']
        parsed['FALSE'] = parsed['embedding_vocab']['FALSE']

        label_vocab = tuple(parsed['embedding_vocab'].items())
        parsed['sorted_entity_labels'] = [x[0] for x in sorted(label_vocab, key=lambda x: x[1])]
        relation_vocab_tuple = tuple(parsed['relation_vocab'].items())
        parsed['sorted_relations'] = [x[0] for x in sorted(relation_vocab_tuple, key=lambda x: x[1])]
        parsed['rPAD'] = parsed['relation_vocab']['PAD']
        parsed['DUMMY_START_RELATION'] = parsed['relation_vocab']['DUMMY_START_RELATION']
        parsed['action_vocab_size'] = len(parsed['relation_vocab'])

        parsed["answer_set"] = json.load(open(parsed['data_input_dir'] + '/answer_set.json'))
        for key,val in parsed.items():
            setattr(self, key, val)

    def asdict(self):
        return vars(self)

if __name__=='__main__':
    Config('config_20200219/config_000.yaml')