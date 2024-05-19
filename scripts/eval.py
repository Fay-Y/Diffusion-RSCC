
from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
from eval_func.spice.spice import Spice
import os
from PIL import Image, ImageDraw, ImageFont
import json
def result_presentation(tgts,hypos,wst,filenames,tokenizer):
    ori_path='data/images/test/D'
    save_path = 'result'
    os.makedirs(save_path, exist_ok=True)
    eval_ref,eval_hypo,tgts,hypos = result_process(tgts,hypos,tokenizer)
    # hypos = wst
    metrics = get_eval_score(eval_ref,eval_hypo)
    
    print('metrics for all the result:',metrics)
    print('img num:',len(filenames))
    output_json = {}
    output_json_raw ={}
    # i = 0
    filenames = filenames
    for i,filename in enumerate(filenames):
        # print('i:',i)
        # print('file name:',filename)
        file_path=os.path.join(ori_path,filename)
        try:
            image = Image.open(file_path)
        except IOError:
            print(f"无法加载图片：{filename}")
            continue

        num_lines = 10  

        font = ImageFont.load_default()  # 默认字体
        line_height = 15
        new_height = image.height + num_lines * line_height
        new_image = Image.new("RGB", (image.width, new_height), "white")
        new_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(new_image)
        initial_position = (5, image.height + 5)  

        # 添加 Reference 文本
        draw.text(initial_position, f"hypo: {hypos[i]}", fill="black", font=font)

        # 添加 GT 文本
        y_offset = initial_position[1]
        for tgt in enumerate(tgts[i]):
            y_offset += line_height          
            draw.text((initial_position[0], y_offset), f"GT: {tgt}", fill="black", font=font)
        
        # print('the mertic result for the image',i)
        # tmp_metric = get_eval_score([eval_ref[i]],[eval_hypo[i]])
        # y_offset+= line_height
        # draw.text((initial_position[0], y_offset), f"bleu4: {tmp_metric['Bleu_4']}", fill="black", font=font)
        # y_offset+= line_height
        # draw.text((initial_position[0], y_offset), f"meteor: {tmp_metric['METEOR']}", fill="black", font=font)
        # y_offset+= line_height
        # draw.text((initial_position[0], y_offset), f"rouge_L: {tmp_metric['ROUGE_L']}", fill="black", font=font)
        # y_offset+= line_height
        # draw.text((initial_position[0], y_offset), f"cider: {tmp_metric['CIDEr']}", fill="black", font=font)

        tmp_save = os.path.join(save_path,filename)

        new_image.save(tmp_save)  
        # output_json[str(i)]={
        #     'filename': filename,
        #     'refs': tgts[i],
        #     'cand': [hypos[i]]
        # }

        sentences = [
        "there is no difference",
        "the two scenes seem identical",
        "the scene is the same as before",
        "no change has occurred",
        "almost nothing has changed"
         ]
        match_hypo =  None
        if hypos[i] in sentences:
            match_hypo = "the scene is the same as before"
        else:
            match_hypo =  hypos[i]
        output_json[str(i)] = {
            'filename':filename,
            'refs': tgts[i],
            'cand': [match_hypo]}
        output_json_raw[str(i)] = {
            'filename':filename,
            'refs': tgts[i],
            'cand': [hypos[i]]}
        
    tmp_json ={}
    tmp_json.update(output_json)   
    json_path = os.path.join(save_path, 'diffusionCC.json') 
    with open(json_path, 'w') as file1:
        json.dump(tmp_json, file1,indent=4)
        
    raw_tmp_json = {}
    raw_tmp_json.update(output_json_raw)
    raw_json_path = os.path.join(save_path, 'diffusionCC_raw.json') 
    with open(raw_json_path, 'w') as file2:
        json.dump(raw_tmp_json, file2,indent=4)

def result_process(references,hypotheses,tokenizer):
    # print('ref:',references)
    eval_ref = []
    references = references
    for reference in references:
        processed_sample = []
        for ref in reference:
            # 从末尾开始找到第一个非零元素的索引
            non_zero_index = len(ref) - 1
            while ref[non_zero_index] != 997:
                non_zero_index -= 1

            processed_row = ref[1:non_zero_index]
            processed_sample.append(processed_row.tolist())
        eval_ref.append(processed_sample)
        
    hypo = hypotheses.indices

    eval_hypo = []
    tmp_hypo = [0,0,0,0,0]
    
    for seq in hypo:
        id = 0
        for item in seq:
            if id == 40 and item != 997:
                eval_hypo.append(tmp_hypo)
                break
            
            if item == 996 and id == 0:
                current_hypo = []
            elif item != 996 and id == 0:
                eval_hypo.append(tmp_hypo)
                break
            elif item == 997:
                eval_hypo.append(current_hypo)
                break
            else:
                current_hypo.extend(item.tolist())
            id += 1
    print('references:',len(eval_ref ))
    print('hypotheses:',len(eval_hypo))
    
    result_ref = []
    for tgts in eval_ref:
        tmp_ref=[]
        for tgt in tgts:
            tokens_r = ' '.join(tokenizer[seq] for seq in tgt)
            tmp_ref.append(tokens_r)
        result_ref.append(tmp_ref)
        
    result_hypo = []
    for hypo in eval_hypo:
        tokens_h = ' '.join(tokenizer[seq] for seq in hypo)
        result_hypo.append(tokens_h)

    # print('hypotheses:',result_hypo)
        
    return  eval_ref, eval_hypo,result_ref,result_hypo

def get_eval_score(references, hypotheses):
  
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]
    
    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]
    #print("hypo",hypo)
    #print("ref",ref)
    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        # print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict