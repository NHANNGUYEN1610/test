
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'



from tensorflow.keras import backend as bk
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
from datetime import datetime
from tqdm import tqdm
from model import modl_mussles 
from load_multiple_models import ImportGraph


import readData as rd 
import misc as sf
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

gpu = get_available_gpus()
print(gpu)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)


tf.reset_default_graph()




#--------------------Set some parameters------------------
nShots=144 # simulate N-Shots data
epochs=300 #number of training epoches
K=1 #number of unrolls
lamK=.05 # regularization parameters for image and k-space

dataset_name='/mnt/server_2/nhannguyen/big_data/mask_espirit_training_data_17_acc_10.npz' #training dataset full file name


#%%Generate a meaningful filename to save the trainined models for testing
print ('*************************************************')
start_time=time.time()
saveDir='trained_model/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%S%P_")+ \
        str(epochs)+'E_'+str(K)+'K_'+str(nShots)+'Shots'+str(lamK)+'Reg'

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName= directory+'/model'

#%% prepare data
X_data, X_true, Y_data, Samp, Mask =rd.getTrnDataNshots(dataset_name)

print(Mask.shape, X_data.shape, Mask.dtype)
nImg=X_true.shape[0]

tf.reset_default_graph()

orgT = tf.placeholder(tf.complex64,shape=(None,nShots,144,192),name='orgT')
dataT  = tf.placeholder(tf.complex64,shape=(None,nShots,144,192),name='dataT')
SampT  = tf.placeholder(tf.complex64,shape=(None,nShots,144,192),name='sampT')
kdataT = tf.placeholder(tf.complex64,shape=(None,nShots,144,192),name='kdataT')
maskT = tf.placeholder(tf.complex64,shape=(None,144,192),name='maskT')



predTst  = modl_mussles(kdataT, dataT, SampT, lamK,K)


sessFileNameTst=directory+'/modelTst'

saver=tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    savedFile=saver.save(sess, sessFileNameTst,latest_filename='checkpointTst')
print ('testing model saved : ' + savedFile)




#%% creating the tensorflow dataset
batchSize=1 
nTrn=X_true.shape[0]
nBatch= int(np.floor(np.float32(nTrn)/batchSize))
nSteps= nBatch*epochs


tf.reset_default_graph()

orgP = tf.placeholder(tf.complex64,shape=(None,nShots,144,192),name='orgP')
dataP  = tf.placeholder(tf.complex64,shape=(None,nShots,144,192),name='dataP')
SampP  = tf.placeholder(tf.complex64,shape=(None,nShots,144,192),name='sampP')
kdataP = tf.placeholder(tf.complex64,shape=(None,nShots,144,192),name='kdataP')
maskP = tf.placeholder(tf.complex64,shape=(None,144,192),name='maskP')


trnData = tf.data.Dataset.from_tensor_slices((dataP, orgP, kdataP, SampP, maskP))

trnData = trnData.cache()
trnData=trnData.repeat(count=epochs)
trnData = trnData.shuffle(buffer_size=5)
trnData=trnData.batch(batchSize)
trnData=trnData.prefetch(5)
iterator=trnData.make_initializable_iterator()
dataT, orgT, kdataT, SampT, maskT = iterator.get_next('getNext')   
## Freeze segmentation graph
def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph

graph_model = load_graph("segmented_model/segment_model.pb")
graph_model_def = graph_model.as_graph_def()

#
def sigmoid_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_pred = tf.abs(y_pred)
    y_true = tf.abs(y_true)
    ce = bk.binary_crossentropy(y_true, y_pred, from_logits=False)
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_factor = y_true *alpha + (1-y_true) * (1-alpha)
    modulating_factor = tf.pow((1.0 - p_t), gamma)
    return tf.reduce_sum(alpha_factor *modulating_factor *ce, axis=-1)

#
segmentation = lambda x : tf.import_graph_def(graph_model_def, input_map={"dataSegT": x},return_elements=["maskSegTst:0"])
#
# # Custom gradient for segmentation network
@tf.custom_gradient
def seg_net(x):
    def grad(dy):
        return tf.zeros_like(x)
    return segmentation(x), grad

# Make training model
with tf.variable_scope('true_model'):
    predT = modl_mussles(kdataT, dataT, SampT, lamK,K)


with tf.variable_scope('seg_net', reuse=tf.AUTO_REUSE):
    pred_maskT = seg_net(predT)


# Define loss functions

loss_weight = 0.
with tf.name_scope('all_losses'):
    with tf.name_scope('img_loss'):
        loss_imgs= tf.reduce_mean(tf.pow(tf.abs(predT-orgT),2))
    with tf.name_scope('mask_loss'):
        loss_mask = sigmoid_focal_crossentropy(maskT, pred_maskT)
    with tf.name_scope('temp_loss'):
        loss_temp =  tf.reduce_mean(tf.abs(sf.tf_TempFFT(predT)-sf.tf_TempFFT(orgT)))
    with tf.name_scope('sum_losses'):
        loss_img = loss_temp * 0. + loss_imgs * 1.
        loss = tf.identity((1. - loss_weight)*loss_img) + loss_weight*loss_mask

tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 15000, 0.75, staircase=True)


all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(all_trainable_vars)

model_vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/unetKsp')
print(model_vars_list)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)



with tf.variable_scope('optimizer'):
    with tf.control_dependencies(update_ops):
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        gvs = optimizer.compute_gradients(loss, var_list=all_trainable_vars)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        opToRun = optimizer.apply_gradients(capped_gvs, global_step=global_step)

#%% training code

print ('*************************************************')
print ('training started at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))

saver = tf.train.Saver(max_to_keep=100)
totalLoss,ep=[],0
lossT = tf.placeholder(tf.float32)
lossSumT = tf.summary.scalar("TrnLoss", lossT)
monitor_loss = []
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    print('Number of trainable parameters: %d' % count_number_trainable_params())
    print(tf.trainable_variables())


    feedDict={orgP:X_true,dataP:X_data,kdataP:Y_data, SampP:Samp, maskP:Mask}
    sess.run(iterator.initializer,feed_dict=feedDict)

    savedFile=saver.save(sess, sessFileName)
    print("Model meta graph saved in::%s" % savedFile)


    writer = tf.summary.FileWriter(directory, sess.graph)

    for step in tqdm(range(nSteps)):
        try:

            tmp, _, _ = sess.run([loss, update_ops, opToRun])
            totalLoss.append(tmp)

            if np.remainder(step+1,nBatch)==0:
                ep=ep+1
                avgTrnLoss=np.mean(totalLoss)

                lossSum=sess.run(lossSumT,feed_dict={lossT:avgTrnLoss})
                writer.add_summary(lossSum,ep)
                totalLoss=[]
 
        except tf.errors.OutOfRangeError:
            break
    writer.close()
    _=saver.save(sess, sessFileName,global_step=ep,write_meta_graph=True)

end_time = time.time()
print ('Trianing completed in minutes ', ((end_time - start_time) / 60))
print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('*************************************************')

