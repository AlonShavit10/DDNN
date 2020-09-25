##############
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def get_total_flops_and_total_parameters(model):
    # model = r"D:\AloNet_skip_Trans_XY\data_set_0\28-08-2020___17-14-59_\checkpoints\full_model.hdf5"


    import keras.backend as K
    # import keras.backend as K
    from keras.applications.mobilenet import MobileNet

    run_meta = tf.compat.v1.RunMetadata()
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.compat.v1.keras.backend.set_session(sess)
        # net = MobileNet(alpha=.75, input_tensor=tf.compat.v1.placeholder('float32', shape=(1,32,32,3)))
        net=tf.keras.models.load_model(model)

        ###
        # newInput = Input(shape=(128*4,1), batch_size=int(1))
        newInput = Input(shape=(28,28, 1), batch_size=int(1))
        newOutputs = net(newInput)
        newModel = Model(newInput, newOutputs)
        ###


        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts =  tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)


        total_flops = flops.total_float_ops
        total_parameters = params.total_parameters


    return total_flops,total_parameters

total_flops,total_parameters = get_total_flops_and_total_parameters(r"C:\temp\selected\train_dependency\conv_train2_test_100_Epoch1000_lr0.01_16_ch_100_44\saved_model\full_model.hdf5")
print("Total flops:")
print("{:,}".format(total_flops))
print("Total parameters:")
print("{:,}".format(total_parameters))