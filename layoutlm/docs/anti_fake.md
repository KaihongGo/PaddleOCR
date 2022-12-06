## 数据格式转化

XFUND -> Paddle


```bash
# 转换标注格式
python3 ppstructure/kie/tools/trans_xfun_data.py --ori_gt_path=train_data/anti_fake/train.json --output_path=train_data/anti_fake/train-paddle.json
```

## 文本检测

```bash

```

## SER

### 训练

```bash
python tools/train.py -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml                                 
```

### 标签映射

```python
def load_vqa_bio_label_maps(label_map_path):
```

### 预测

```bash
python3 tools/infer_kie_token_ser.py \
  -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml \
  -o Architecture.Backbone.checkpoints=./output/ser_vi_layoutxlm_xfund_zh/best_accuracy \
  Global.infer_img=./ppstructure/docs/kie/input/zh_val_42.jpg
```

预测图片如下所示，图片会存储在`Global.save_res_path`路径中。


预测过程中，默认会加载PP-OCRv3的检测识别模型，用于OCR的信息抽取，如果希望加载预先获取的OCR结果，可以使用下面的方式进行预测，指定`Global.infer_img`为标注文件，其中包含图片路径以及OCR信息，同时指定`Global.infer_mode`为`False`，表示此时不使用OCR预测引擎。

```bash
python3 tools/infer_kie_token_ser.py \
  -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml \
  -o Architecture.Backbone.checkpoints=./output/ser_vi_layoutxlm_xfund_zh/best_accuracy \
  Global.infer_img=./train_data/XFUND/zh_val/val.json \
  Global.infer_mode=False
```

```
python3 tools/infer_kie_token_ser.py \
  -c output/anti_fake/experiments/ser_vi_layoutxlm_xfund_zh/config.yml \
  -o Architecture.Backbone.checkpoints=output/anti_fake/experiments/ser_vi_layoutxlm_xfund_zh/best_accuracy \
  Global.infer_img=train_data/anti_fake/train-paddle.json \
  Global.infer_mode=False
```

## det

修改配置文件configs/det/ch_PP-OCRv2_det_student.yml中的以下字段：

```
Global.pretrained_model：指向预训练模型路径
Train.dataset.data_dir：指向训练集图片存放目录
Train.dataset.label_file_list：指向训练集标注文件
Eval.dataset.data_dir：指向验证集图片存放目录
Eval.dataset.label_file_list：指向验证集标注文件
Optimizer.lr.learning_rate：调整学习率，本实验设置为0.005
Train.dataset.transforms.EastRandomCropData.size：训练尺寸改为[1600, 1600]
Eval.dataset.transforms.DetResizeForTest：评估尺寸，添加如下参数
       limit_side_len: 1600
       limit_type: 'min'
```

注意：在使用上述预训练模型的时候，需要使用文件夹中的student.pdparams文件作为预训练模型，即，仅使用学生模型。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
        -c configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_student.yml
```

导出

```bash
python tools/export_model.py \
    -c output/anti_fake/exp1/ch_PP-OCR_V3_det/config.yml \
    -o Global.pretrained_model="output/anti_fake/exp1/ch_PP-OCR_V3_det/best_accuracy" \
    Global.save_inference_dir="output/anti_fake/exp1/ch_PP-OCR_V3_det/det_db_inference/"
```

推理

```
# det_algorithm="DB"
python tools/infer/predict_det.py \
    --det_model_dir="output/anti_fake/exp1/ch_PP-OCR_V3_det/det_db_inference" \
    --image_dir="./doc/vqa/input/zh_val_21.jpg" \
    --use_gpu=True
```

## RE


### 训练

```
python3 ./tools/infer_kie_token_ser_re.py \
-c configs/kie/vi_layoutxlm/re_vi_layoutxlm_xfund_zh.yml \
-o Architecture.Backbone.checkpoints=./pretrain_models/re_vi_layoutxlm_udml_xfund_zh/re_layoutxlm_xfund_zh_v4_udml/best_accuracy/ \
Global.infer_img=./train_data/XFUND/zh_val/val.json \
Global.infer_mode=False \
-c_ser configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml \
-o_ser Architecture.Backbone.checkpoints=pretrain_models/ser_vi_layoutxlm_udml_xfund_zh/best_accuracy/
```

### 推理模型

```bash
cd ppstructure
python3 kie/predict_kie_token_ser_re.py \
  --kie_algorithm=LayoutXLM \
  --re_model_dir=../inference/re_vi_layoutxlm \
  --ser_model_dir=../inference/ser_vi_layoutxlm \
  --use_visual_backbone=False \
  --image_dir=./docs/kie/input/zh_val_42.jpg \
  --ser_dict_path=../train_data/XFUND/class_list_xfun.txt \
  --vis_font_path=../doc/fonts/simfang.ttf \
  --ocr_order_method="tb-yx"
```