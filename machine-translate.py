from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import zhconv
from torchmetrics import BLEUScore

device = torch.device("cuda:1")
"""
zho_Hans # 汉语
jpn_Jpan # 日语
eng_Latn # 英语
rus_Cyrl # 俄语
"""

tokenizer = AutoTokenizer.from_pretrained("/home/lv/nllb-200/nllb")
model = AutoModelForSeq2SeqLM.from_pretrained("/home/lv/nllb-200/nllb").to(device)

translator = pipeline(
    'translation',
    model=model,
    tokenizer=tokenizer,
    src_lang='zho_Hans',
    tgt_lang='eng_Latn',
    max_length=512
)

filepath = "/home/lv/nllb-200/nllb/dataset/zh-en.tsv"

with open(filepath, "r", encoding="utf-8") as f:
    filedata = f.readlines()
    samedict = dict()  # 相同输入数据的index
    for index, linedata in enumerate(filedata):
        id = int(linedata.strip().split("\t")[0].replace("\ufeff", ""))
        if id in samedict:
            samedict[id].append(index)
        else:
            samedict[id] = [index]

    # 遍历相同的数据进行计算
    #targetlist = []
    total = 0
    count = 0
    for key in samedict.keys():
        count = count + 1 
        targetlist = []
        inputdata = filedata[samedict[key][0]].split("\t")[1].strip()
        inputdata = zhconv.convert(inputdata, "zh-hans")
        translate_output = translator(inputdata)
        for i in samedict[key]:
            targetlist.append(filedata[i].split("\t")[3].strip().split())
        targetlist = [targetlist]
        print("标签内容：",targetlist)
        print("识别结果:",translate_output)

        translate_input = translate_output[0]["translation_text"]

        translate_input = [translate_input.split()]

        bleu_4 = BLEUScore(n_gram=1)

        print("识别得分:",bleu_4(targetlist, translate_input))
        total = total + bleu_4(targetlist, translate_input)
    print(total/count)
