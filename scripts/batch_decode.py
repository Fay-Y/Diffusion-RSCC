import os, sys, glob
# full_lst = glob.glob('diff_models_synth128*')
# full_lst = glob.glob('diff_models_synth32*')
# full_lst = glob.glob('diff_models_synth32_3_rand16*')
# full_lst = glob.glob('diff_models_synth_rand_16_trans_lr_1e-5_long_Lsimple')
print("sys.argv[1]:",sys.argv[1])
full_lst = glob.glob(sys.argv[1])
print("full_list:",full_lst)
top_p = -1.0 if len(sys.argv) < 2 else sys.argv[2]
print(f'top_p = {top_p}')
pattern_ = 'model' if len(sys.argv) < 3 else sys.argv[3]
print(f'pattern_ = {pattern_}', sys.argv[3])


output_lst = []
for lst in full_lst:
    print("lst:",lst)
    try:

        tgt = sorted(glob.glob(f"{lst}/{pattern_}*pt"))[-1]

        lst = os.path.split(lst)[1]
        print("tgt:", tgt)
        num = 1
    except:
        continue
    model_arch_ = lst.split('_')[5-num]
    #model_arch_ = 'transformer'
    model_arch = 'conv-unet' if 'conv-unet' in lst else 'transformer'
    mode =  'image' if ('conv' in model_arch ) else 'text' #or '1d-unet' in model_arch_
    print("mode:",mode,"model_arch:", model_arch)
    dim_ =lst.split('_')[4-num]
    #print('dim:',dim)
    #dim_ ='16'
    # diffusion_steps= 4000
    # noise_schedule = 'cosine'
    # dim = dim_.split('rand')[1]

    if 'synth' in lst:
        modality = 'synth'
    elif 'pos' in lst:
        modality = 'pos'
    elif 'image' in lst:
        modality = 'image'
    elif 'roc' in lst:
        modality = 'roc'
    elif 'e2e-tgt' in lst:
        modality = 'e2e-tgt'
    elif 'simple-wiki' in lst:
        modality = 'simple-wiki'
    elif 'book' in lst:
        modality = 'book'
    elif 'yelp' in lst:
        modality = 'yelp'
    elif 'commonGen' in lst:
        modality = 'commonGen'
    elif 'e2e' in lst:
        modality = 'e2e'

    modality = 'e2e-tgt'
    if 'synth32' in lst:
        kk = 32
    elif 'synth128' in lst:
        kk = 128

    try:
        diffusion_steps = int(lst.split('_')[7-num])
       
    except:
        diffusion_steps = 4000
    #diffusion_steps = 500
    print("diffusion_steps:",diffusion_steps)   
    try:
        noise_schedule = lst.split('_')[8-num]
        assert  noise_schedule in ['cosine', 'linear']
        print(noise_schedule)
    except:
        noise_schedule = 'cosine'
    try:
        dim = int(dim_.split('rand')[1])
        #dim = 16
    except:
        dim =lst.split('_')[4-num]
    print('dim:',dim)
    try:
        print(len(lst.split('_')))
        num_channels =  int(lst.split('_')[-1].split('h')[1])
    except:
        num_channels = 128

    print(tgt, model_arch, dim, num_channels)
    # out_dir = 'diffusion_lm/improved_diffusion/out_gen_large_nucleus'
    # num_samples = 512

    # out_dir = 'diffusion_lm/improved_diffusion/out_gen_v2_nucleus'

    out_dir = 'generation_outputs'
    num_samples = 1929

    if modality == 'e2e':
        num_samples = 547
    print('modality:',modality)

    COMMAND = f'python scripts/{mode}_sample.py ' \
    f'--model_path {tgt} --batch_size 64 --num_samples {num_samples} --top_p {top_p} ' \
    f'--out_dir {out_dir} '
    print(COMMAND)
    # os.system(COMMAND)

    # shape_str = "x".join([str(x) for x in arr.shape])
    model_base_name = os.path.basename(os.path.split(tgt)[0]) + f'.{os.path.split(tgt)[1]}'
    print('model_base_name:',model_base_name)
    if modality == 'e2e-tgt' or modality == 'e2e':
        out_path2 = os.path.join(out_dir, f"{model_base_name}.samples_{top_p}.json")
        
    else:
        out_path2 =  os.path.join(out_dir, f"{model_base_name}.samples_{top_p}.txt")
    output_cands = glob.glob(out_path2)
    print("out_path2:",out_path2,"out_cands:", output_cands)
    if len(output_cands) > 0:
        out_path2 = glob.glob(out_path2)[0]
    else:
        os.system(COMMAND)
        out_path2 = glob.glob(out_path2)[0]

    output_lst.append(out_path2)

    # if modality == 'pos':
    #     model_name_path = 'predictability/diff_models/pos_e=15_b=20_m=gpt2_wikitext-103-raw-v1_s=102'
    # elif modality == 'synth':
    #     if kk == 128:
    #         model_name_path = 'predictability/diff_models/synth_e=15_b=10_m=gpt2_wikitext-103-raw-v1_None'
    #     else:
    #         model_name_path = 'predictability/diff_models/synth_e=15_b=20_m=gpt2_wikitext-103-raw-v1_None'
    # elif modality == 'e2e-tgt':
    #     model_name_path = "predictability/diff_models/e2e-tgt_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_None"
    #     #model_name_path="/root/yxf/Diffusion_New_test/diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e"
    # elif modality == 'roc':
    #     model_name_path = "predictability/diff_models/roc_e=6_b=10_m=gpt2_wikitext-103-raw-v1_101_wp_pad_v1"
    # elif modality == 'e2e':
    #     COMMAND1 = f"python diffusion_lm/e2e_data/mbr.py {out_path2}"

    #     os.system(COMMAND1)
    #     COMMAND2 = f"python e2e-metrics/measure_scores.py " \
    #                f"diffusion_lm/improved_diffusion/out_gen_v2_dropout2/1_valid_gold  " \
    #                f"{out_path2}.clean -p  -t -H > {os.path.join(os.path.split(tgt)[0], 'e2e_valid_eval.txt')}"
    #     print(COMMAND2)
    #     os.system(COMMAND2)
    #     continue
    # else:
    #     print('not trained a AR model yet... only look at the output plz.')
    #     continue
    # COMMAND = f"python scripts/ppl_under_ar.py " \
    #           f"--model_path {tgt} " \
    #           f"--modality {modality}  --experiment random " \
    #           f"--model_name_or_path {model_name_path} " \
    #           f"--input_text {out_path2}  --mode eval"

    print(COMMAND)
    print()
    # os.system(COMMAND)
print('output lists:')
print("\n".join(output_lst))