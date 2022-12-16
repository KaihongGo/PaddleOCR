第一阶段：VQATokenLabelEncode

ocr_infos: 原始OCR块一维化并排序

entities: 分词后 [{entity}, {entity}]
    entity: {
        "start": 开始token的index,
        "end": 结束token的index,
        "label": 实体类型,
    }
    一个OCR块是一个实体

relations: [(_, _), (_, _)]
    _ : (实体1的ocr_infos index, 实体2的ocr_infos index)
    两个实体之间的关系

id2label: {id: label}
    id: 实体ocr_infos index
    label: 实体类型

empty_entity: []
    空实体的ocr_infos index

entity_id_to_index_map: {id: index}
    id: 实体ocr_infos index
    index: 实体在entities中的index


第二阶段 VQAReTokenRelation

> 这个阶段可以和第一个阶段合并


删除 id2label, empty_entity, entity_id_to_index_map

relations: [{relation}, {relation}] 更新
    relation: {
        "head": entities中的index,
        "tail": entities中的index,
        "start_index": 开始token的index,
        "end_index": 结束token的index,
    }

第三阶段 VQAReTokenChunk （似乎这个阶段不起作用）

tokens 可以被分成 多个 [chunk]，每个chunk是 max_seq_length 长度的tokens
约束实体在一个chunk中，关系在一个chunk中

实体标签映射为 数值，

问题：这里仅仅考虑第一个 chunk，后面的chunk怎么处理？

item: {
    "entities": [entity, entity],
    "relations": [relation, relation],
}

entity: {
    "start": 开始token的index,
    "end": 结束token的index,
    "label": 实体类型,
}

relation: {
    "head": entities中的index,
    "tail": entities中的index,
    "start_index": 开始token的index,
    "end_index": 结束token的index,
}

虽然处理了全部，实际上相当于截断后面的，只考虑前面的;

该阶段删除 entities, relations


第四阶段 TensorizeEntitiesRelations

转换为向量 

data: {
    "entities": [entity, entity],
    "relations": [relation, relation],
}

最后 

entities: 

shape = [max_seq_len + 1, 3]

第一行: [start, end, label] 长度
第二行至最后一行: [start, end, label] 数值，每行一个实体
[开始token的index, 结束token的index, 实体类型]

relations:

shape = [max_seq_len*max_seq_len + 1, 2]

第一行: [head, tail] 长度
第二行至最后一行: [head, tail] 数值，每行一个关系
[实体1的entities中的index, 实体2的entities中的index]