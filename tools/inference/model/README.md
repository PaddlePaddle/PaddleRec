# 文件说明
1. rec_inference.pdmodel、rec_inference.pdiparams：完整的模型和参数文件
2. pruned_inference.pdmodel、pruned_inference.pdiparams：使用 cube 服务时所使用的裁剪后的模型和参数文件，相比 1，少了 lookup_table 层。
3. rec_inference.zip：java 推理所使用的模型和参数文件压缩包
