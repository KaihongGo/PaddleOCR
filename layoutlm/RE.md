## RE

输入

```python
return ser_inputs, entity_idx_dict_batch
# ser_inputs:
'input_ids', 'bbox', 'attention_mask', 
'token_type_ids', 'image', entities, relations
```

输出

```python
# 获得第几个实体的ocr_info
ocr_info_head = ser_result[entity_idx_dict[relation['head_id']]] 
ocr_info_tail = ser_result[entity_idx_dict[relation['tail_id']]]
result.append((ocr_info_head, ocr_info_tail))
```

如果 `infer_mode is False`，那么会加载文件中的 `ocr_info`
输入 `data` 变为

```python
# {
#     "img_path": img_path,
#     "label": label,
# }

if config["Global"].get("infer_mode", None) is False:
    data_line = info.decode('utf-8')
    substr = data_line.strip("\n").split("\t")
    img_path = os.path.join(data_dir, substr[0])
    data = {'img_path': img_path, 'label': substr[1]}
```

`SER` 模型 `VQATokenLabelEncode` 预处理变换过程中，会判断 `infer_mod` 的状态，从而决定是否调用 OCR 引擎，生成 `ocr_info` 信息。

后处理 位于 `VQAReTokenLayoutLMPostProcess` 中

```python
# merge relations and ocr info
results = []
for pred_relation, ser_result, entity_idx_dict in zip(
        pred_relations, ser_results, entity_idx_dict_batch):
    result = []
    used_tail_id = []
    for relation in pred_relation:
        # 尾端只有一个
        if relation['tail_id'] in used_tail_id:
            continue
        used_tail_id.append(relation['tail_id'])
        ocr_info_head = ser_result[entity_idx_dict[relation['head_id']]]
        ocr_info_tail = ser_result[entity_idx_dict[relation['tail_id']]]
        result.append((ocr_info_head, ocr_info_tail))
    results.append(result)
return results
```

### 训练标签


## 模型

#### RE

head: [xxx, 实体标签（类别）]; tail: [xxx, 实体标签]

xxx是实体的嵌入表示？
但是标注的时候实体可能被划分为多个token，然而模型采用 LayoutLM 的输出，仅仅索引实体的start -token的嵌入表示？
**问题**：是否可以 start->end 取平均/映射 表示实体的嵌入表示！

输入:

[[head], [tail]]
头尾实体嵌入表示，模型接收两个输入，采用线性/双线性映射，得到实体关系的 [all_possible_relations, 2] 的分类结果。实际关系类别只有一类，所以分类为一类！

**思考：**NLP怎么做关系分类

all_possible_relations 怎么构造正负样本对。
positive: 输入的数据
negative: 除去输入的

几个重要索引

ocr_infos = [] 索引为实体顺序，不会改变
entities = [id, id2] # 存储实体的索引位置，在ocr_infos 位置


## 训练

### 数据预处理

```python
# keep_keys =
['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'entities', 'relations']
```

预处理

`VQAReTokenChunk` 对实体进行改变

[{start:int, end:int, label:int},{start:int, end:int, label:int}] -> {start:[], end:[], label:[]}

`TensorizeEntitiesRelations`

将实体变成了

[[len(start), start...], [len(end), end...], [len(label), label...]]

## 推理

利用已有OCR结果

```
python3 tools/infer_kie_token_ser_re.py \
-c output/union/exp/re_vi_layoutxlm_xfund_zh/config.yml \
-o Architecture.Backbone.checkpoints=output/union/exp/re_vi_layoutxlm_xfund_zh/best_accuracy \
Global.save_res_path=output/union/val/result_ocr \
Global.infer_img=train_data/union/val.json \
Global.infer_mode=False -c_ser output/union/exp/ser_vi_layoutxlm_xfund_zh/config.yml \
-o_ser Architecture.Backbone.checkpoints=output/union/exp/ser_vi_layoutxlm_xfund_zh/best_accuracy
```

调用OCR引擎

```
python3 tools/infer_kie_token_ser_re.py \
-c output/union/exp/re_vi_layoutxlm_xfund_zh/config.yml \
-o Architecture.Backbone.checkpoints=output/union/exp/re_vi_layoutxlm_xfund_zh/best_accuracy \
Global.save_res_path=output/union/val/result \
Global.infer_img=train_data/union/image \
Global.infer_mode=True \
-c_ser output/union/exp/ser_vi_layoutxlm_xfund_zh/config.yml \
-o_ser Architecture.Backbone.checkpoints=output/union/exp/ser_vi_layoutxlm_xfund_zh/best_accuracy
```

output/union/val/result_ocr/TW-K12-1LHS601PO-001_page-83_ser_re.jpg

```
python3 tools/infer_kie_token_ser_re.py \
-c output/union/exp/re_vi_layoutxlm_xfund_zh/config.yml \
-o Architecture.Backbone.checkpoints=output/union/exp/re_vi_layoutxlm_xfund_zh/best_accuracy \
Global.save_res_path=output/union/val/result \
Global.infer_img=train_data/union/image/TW-K12-1LHS601PO-001_page-83_ser_re.png \
Global.infer_mode=True \
-c_ser output/union/exp/ser_vi_layoutxlm_xfund_zh/config.yml \
-o_ser Architecture.Backbone.checkpoints=output/union/exp/ser_vi_layoutxlm_xfund_zh/best_accuracy
```


SER

with ocr

```bash
python3 tools/infer_kie_token_ser.py \
-c output/union/exp/ser_vi_layoutxlm_xfund_zh/config.yml \
-o Architecture.Backbone.checkpoints=output/union/exp/ser_vi_layoutxlm_xfund_zh/best_accuracy \
Global.infer_img=train_data/union/val.json \
Global.infer_mode=False \
Global.save_res_path=output/union/result/result_ocr
```

```bash
python3 tools/infer_kie_token_ser.py \
-c output/union/exp/ser_vi_layoutxlm_xfund_zh/config.yml \
-o Architecture.Backbone.checkpoints=output/union/exp/ser_vi_layoutxlm_xfund_zh/best_accuracy \
Global.infer_img=train_data/union/image \
Global.infer_mode=True \
Global.save_res_path=output/union/result/result
```