# -*- coding:utf-8 -*-

import tensorflow as tf

model_path = 'your/model/path'
model_version = '4'
export_model_dir = 'output/path' + model_version


graph = tf.Graph()
with graph.as_default():
    # 导入你已经训练好的模型
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        checkpoint_file = tf.train.latest_checkpoint(model_path)
        print(checkpoint_file)
        print('Restore from {}'.format(model_path))
        saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
        saver.restore(sess, checkpoint_file)

        input_left = graph.get_tensor_by_name('input_left:0')  # [batch, max_len]
        input_right = graph.get_tensor_by_name('input_right:0')
        keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
        y_pred_cls = graph.get_tensor_by_name('output/scores:0')  # [batch, 2]

        # 定义导出模型的各项参数
        # 定义导出地址
        print('Exporting trained model to', export_model_dir)
        builder = tf.saved_model.builder.SavedModelBuilder(export_model_dir)

        # 定义Input tensor info
        inputs = {
            'input_left': tf.saved_model.utils.build_tensor_info(input_left),
            'input_right': tf.saved_model.utils.build_tensor_info(input_right),
            'keep_prob': tf.saved_model.utils.build_tensor_info(keep_prob)
        }

        # 定义Output tensor info
        tensor_info_output = tf.saved_model.utils.build_tensor_info(y_pred_cls)

        # 创建预测签名
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs={'predict': tensor_info_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'text_matching_tf_serving': prediction_signature})

        # 导出模型
        builder.save()
        print('Done exporting!')

