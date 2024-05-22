import json
from collections import Counter,defaultdict
import os
import torch.nn
    
def word_processor(rawfile_path ='/Diffusion-RSCC/data/LevirCCcaptions.json',output_folder ='/Diffusion-RSCC/datasets'):

    # 读取raw caption文件
    with open(rawfile_path, 'r') as raw_file:
        data = json.load(raw_file)
        
    split_file_raw_dict = defaultdict(dict)

    max_sentence_length = 0
    word_freq = Counter() 
    for item in data['images']:
        captions = []
        for sentence in item['sentences']:
            tokens = sentence['tokens']
            word_freq.update(tokens)
            # if len(tokens) == 39:
            #     print('tokens 39:',tokens)
            max_sentence_length = max(max_sentence_length, len(tokens))
            captions.append(tokens)
        if len(captions) == 0:
            continue
        
    print('max caption length:',max_sentence_length)   
    max_sentence_length += 2
    print('final max sentence length(with<start>,<end>)',max_sentence_length)   
    words = [w for w in word_freq.keys()]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0


        # Save word map to a JSON
    with open(os.path.join(output_folder, 'vocab' + '.json'), 'w') as j:
            json.dump(word_map, j)
            

    for item in data['images']:
        filename = item['filename']
        split = item['split']
        sentences = item['sentences']
        raw_list = []

        for sentence in sentences:
            tokens = sentence['tokens']
            tokens = ['<start>'] + tokens + ['<end>']  
            encoded_tokens = [word_map[word] for word in tokens]
            padded_encoded_tokens = encoded_tokens + [0] * (max_sentence_length - len(encoded_tokens))
            
            raw_list.append(padded_encoded_tokens)
        split_file_raw_dict[split][filename] = raw_list

    for split, file_raw_dict in split_file_raw_dict.items():
        output_filename = os.path.join(output_folder,f'{split}_caption_encoded.json')
        with open(output_filename, 'w') as output_file:
            json.dump(file_raw_dict, output_file, separators=(',', ':'))
    in_channel = 16
    model = torch.nn.Embedding(len(word_map), in_channel)
    print('initializing the random embeddings', model)
    torch.nn.init.normal_(model.weight)
    path_save = f'datasets/random_emb.torch'
    print(f'save the random encoder to datasets/random_emb.torch')
    torch.save(model.state_dict(), path_save)
word_processor()

