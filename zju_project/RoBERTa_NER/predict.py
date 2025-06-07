from transformers import BertTokenizerFast, AutoModelForTokenClassification
import torch
import numpy as np
import os
from pathlib import Path

os.environ['TRANSFORMERS_OFFLINE']="1"

model_name = "bert_ner"  # ner bert
model_path = os.path.join(Path(__file__).parent.resolve(), "BERT_NER")
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

def predict(text):
    inputs = tokenizer(
        list(text),  # 按字分割输入
        is_split_into_words=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = np.argmax(logits.detach().numpy(), axis=2)
    # 计算置信度（通过Softmax）
    probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]  # [seq_len, num_labels]

    # 提取实体
    word_ids = inputs.word_ids()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    current_entity = []
    entities = []
    weights = []
    confidences = []
    start_idx = 0
    end_idx = 0
    for i, (word_idx, pred) in enumerate(zip(word_ids, predictions[0])):
        if word_idx is None:
            continue
        label = model.config.id2label[pred]
        weight = 1.0 if label != "O" else 0.2
        confidence = probabilities[word_idx][pred]
        weights.append(weight)
        confidences.append(confidence)
        if label == "O":
            if current_entity:
                end_idx = word_idx + 1
                entities.append({
                    'start': start_idx,
                    'end': end_idx,
                    'text': "".join(current_entity), 
                    'labels': [entity_type],
                })
                current_entity = []
        else:
            entity_type = label.split("-")[1]
            if label.startswith("B-"):
                if current_entity:
                    entities.append(("".join(current_entity), entity_type))
                current_entity = [tokens[i]]
                start_idx = word_idx
            else:
                current_entity.append(tokens[i])
    if current_entity:
        entities.append({
            'start': start_idx,
            'end': len(text),
            'text': "".join(current_entity), 
            'labels': [entity_type],
        })
    weighted_sum = np.sum([c*w for c,w in zip(confidences, weights)])
    total_weight = np.sum(weights)
    confidence_total = weighted_sum / total_weight
    print("Confidence Score: ", confidence_total)
    print("Entities:", entities)
    # 输出示例：Entities: [('王小明', 'NAME'), ('北京大学', 'ORG')]
    results = []
    for entity in entities:
        results.append({
            "value": entity,
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
        })
    preds = [{
        "model_version": model_name,
        "score": confidence_total,
        "result": results
    }]
    return preds

def predict_demo(text):
    if text.startswith("患者"):
        results = [{
            "model_version": "DeepSeek-R1-671B",
            "score": 0.92,
            "result": [
                {
                "value": {
                    "start": 18,
                    "end": 20,
                    "text": "挂号",
                    "labels": [
                    "功能性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 9,
                    "end": 15,
                    "text": "在线预约系统",
                    "labels": [
                    "可操作性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 29,
                    "end": 35,
                    "text": "缩短排队时间",
                    "labels": [
                    "非功能性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 47,
                    "end": 54,
                    "text": "扫描电子健康码",
                    "labels": [
                    "可操作性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 56,
                    "end": 61,
                    "text": "打印检查单",
                    "labels": [
                    "功能性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 71,
                    "end": 79,
                    "text": "实时结果推送服务",
                    "labels": [
                    "非功能性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 93,
                    "end": 99,
                    "text": "扫码支付药费",
                    "labels": [
                    "可操作性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 104,
                    "end": 111,
                    "text": "核对药品有效期",
                    "labels": [
                    "功能性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 118,
                    "end": 121,
                    "text": "安全性",
                    "labels": [
                    "非功能性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                }
            ]
        }]
    else:
        results = [{
            "model_version": "DeepSeek-R1-671B",
            "score": 0.98,
            "result": [
                {
                "value": {
                    "start": 4,
                    "end": 10,
                    "text": "语音导航设备",
                    "labels": [
                    "可操作性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 24,
                    "end": 30,
                    "text": "处理外伤出血",
                    "labels": [
                    "功能性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 38,
                    "end": 47,
                    "text": "保持操作过程透明化",
                    "labels": [
                    "非功能性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 57,
                    "end": 68,
                    "text": "医院APP同步电子病历",
                    "labels": [
                    "可操作性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 76,
                    "end": 82,
                    "text": "联合会诊申请",
                    "labels": [
                    "功能性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 89,
                    "end": 99,
                    "text": "每日费用明细自动生成",
                    "labels": [
                    "非功能性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 117,
                    "end": 124,
                    "text": "智能导诊机器人",
                    "labels": [
                    "可操作性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 127,
                    "end": 132,
                    "text": "满意度调研",
                    "labels": [
                    "功能性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                },
                {
                "value": {
                    "start": 140,
                    "end": 147,
                    "text": "指引标识清晰度",
                    "labels": [
                    "非功能性目标"
                    ]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                }
            ]
        }]
    return results

if __name__ == "__main__":
    text_ = "患者通过医院官网的在线预约系统完成了挂号，希望避开早高峰以缩短排队时间。就诊当天，他在自助机上扫描电子健康码快速打印检查单，同时要求检验科提供实时结果推送服务避免反复跑动。取药时，他选择扫码支付药费并强调需要核对药品有效期，以确保用药的安全性。"
    # print(predict(text_))
    print(predict_demo(text_))