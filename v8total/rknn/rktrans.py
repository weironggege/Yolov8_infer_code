from rknn.api import RKNN

# 初始化RKNN对象
rknn = RKNN(verbose=True)

# 配置模型
print('--> Config model')
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3568')

# 加载PyTorch模型
print('--> Loading model')
model_path = 'yolov8n.onnx'  # 替换为你的YOLOv8n模型路径
input_size_list = [[3, 640, 640]]  # 根据你的输入尺寸调整
rknn.load_onnx(model=model_path)

# 构建模型
print('--> Building model')
rknn.build(do_quantization=False)  # 可以选择是否进行量化

# 导出RKNN模型
print('--> Export RKNN model')
output_model_path = 'yolov8n.rknn'  # 输出RKNN模型的路径
rknn.export_rknn(output_model_path)

# 释放资源
rknn.release()

print('Done')
