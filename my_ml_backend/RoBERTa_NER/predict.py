from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "hfl/chinese-roberta-wwm-ext"  # 哈工大RoBERTa
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(
#     model_name,
#     num_labels=num_labels,  # 标签数量（根据任务调整）
#     id2label=id2label,      # 标签ID到名称的映射
#     label2id=label2id
# )

def predict(text):
    # inputs = tokenizer(
    #     list(text),  # 按字分割输入
    #     is_split_into_words=True,
    #     return_tensors="pt"
    # )
    # outputs = model(**inputs)
    # predictions = np.argmax(outputs.logits.detach().numpy(), axis=2)

    # # 提取实体
    # word_ids = inputs.word_ids()
    # tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # current_entity = []
    # entities = []
    # for i, (word_idx, pred) in enumerate(zip(word_ids, predictions[0])):
    #     if word_idx is None:
    #         continue
    #     label = id2label[pred]
    #     if label == "O":
    #         if current_entity:
    #             entities.append(("".join(current_entity), entity_type))
    #             current_entity = []
    #     else:
    #         entity_type = label.split("-")[1]
    #         if label.startswith("B-"):
    #             if current_entity:
    #                 entities.append(("".join(current_entity), entity_type))
    #             current_entity = [tokens[i]]
    #         else:
    #             current_entity.append(tokens[i])
    # if current_entity:
    #     entities.append(("".join(current_entity), entity_type))

    # print("Entities:", entities)
    # 输出示例：Entities: [('王小明', 'NAME'), ('北京大学', 'ORG')]
    results = [{
        "model_version": model_name,
        "score": 0.92,
        "result": [
            {
                "value": {
                    'start': 2, 
                    'end': 6, 
                    'text': '忘记评价', 
                    'labels': ['人物']
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels"
            },
            {
                'value': {
                    'start': 8, 
                    'end': 11, 
                    'text': '大大的', 
                    'labels': ['组织']
                }, 
                'from_name': 'label', 
                'to_name': 'text', 
                'type': 'labels'
            },
        ]
    }]
    return results