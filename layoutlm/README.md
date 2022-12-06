# README

## 文档

[PaddleOCR/发票关键信息抽取.md at release/2.6 · PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/applications/%E5%8F%91%E7%A5%A8%E5%85%B3%E9%94%AE%E4%BF%A1%E6%81%AF%E6%8A%BD%E5%8F%96.md)

## 训练

[PaddleOCR/kie.md at release/2.6 · PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/kie.md)



### 开始训练

## 预测

```bash
python3 tools/infer_kie_token_ser.py \
  -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml \
  -o Architecture.Backbone.checkpoints=./pretrained_model/ser_vi_layoutxlm_xfund_pretrained/best_accuracy \
  Global.infer_img=./ppstructure/docs/kie/input/zh_val_42.jpg
```

### 视觉嵌入

视觉嵌入

```python
visual_embeddings = self.visual_proj(self.visual(image.astype(paddle.float32)))
embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
```

输入图片: [batch_size, 3, 224, 224]
输出：[batch_size, 7*7, 768]

```
class VisualBackbone(nn.Layer)
# 全图卷积: [batch_size, channel, 56, 56]
# 池化: [batch_size, channel, 7, 7]]
# flatten(start_axis=2): [batch_size, channel, 7*7] 
# transpose: [batch_size, 7*7, channel] # channel=256

# 线性投影: [batch_size, 7*7, 768]

# config["image_feature_pool_shape"] = [7, 7, 256]
--> [7, 7, 768]
```

**改进思路**

给文本嵌入拼接上视觉嵌入 --> 768维度其中一部分是该单元格的视觉嵌入

###  输出

```python
return sequence_output, pooled_output, encoder_outputs[1]
# encoder_outputs: (tensor, dict)
# tensor(sequence_output) --> pooled_output: [batch_size, 768]
# dict_keys(['input_hidden_states', 'input_attention_mask', 'input_layer_head_mask', '0_data', '1_data', '2_data', '3_data', '4_data', '5_data', '6_data', '7_data', '8_data', '9_data', '10_data', '11_data'])
```
