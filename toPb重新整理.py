

from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import os
import os.path as osp
from tensorflow.keras import backend as K


sess = tf1.Session()
with tf1.io.gfile.GFile(os.path.join(os.getcwd(), "logs", "model.pb"), "rb") as f:
    graph_def = tf1.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf1.import_graph_def(graph_def, name='')

init = tf1.global_variables_initializer()
sess.run(init)
#input
input_image=sess.graph.get_tensor_by_name("input_image:0")
input_image_meta=sess.graph.get_tensor_by_name("input_image_meta:0")
input_anchors=sess.graph.get_tensor_by_name("input_anchors:0")
#output
detection=sess.graph.get_tensor_by_name("output:0")

inputs = tf.keras.Input(shape=(100,),dtype='float32', name='sample', sparse=True ,batch_size=32)
print(inputs.shape)
# output: (32, 100)

#路径参数
input_path = 'D:\\学习\\高光谱项目\\to_lin\\to_lin\\'
weight_file = 'H5_model.h5'
weight_file_path = osp.join(input_path,weight_file)
output_graph_name = weight_file[:-3] + '.pb'
#转换函数
def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = False):
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i],out_prefix + str(i + 1))
    sess = K.get_session()
    from tensorflow.python.framework import graph_util,graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir,model_name),output_dir)
#输出路径
output_dir = osp.join(os.getcwd(),"D:\\学习\\高光谱项目\\to_lin\\to_lin\\")
#加载模型
h5_model = load_model(weight_file_path)
h5_to_pb(h5_model,output_dir = output_dir,model_name = output_graph_name)
print('model saved')