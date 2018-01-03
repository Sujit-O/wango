# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""To perform inference on test set given a trained model."""
from __future__ import print_function

import codecs
import tensorflow as tf
import attention_model
import gnmt_model
import model as nmt_model
import model_helper

from utils import misc_utils as utils



def decode_inference_indices(model, sess):
     nmt_outputs, infer_summary = model.decode(sess)
     assert nmt_outputs.shape[0] == 1
     
#     if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
     # Select a sentence
     output = nmt_outputs[0, :].tolist()
#
#     # If there is an eos symbol in outputs, cut them at that point.
     if "</s>" in output:
        output = output[:output.index("</s>")]
     
     translation = utils.format_text(output)
    
     return translation
  


def load_data(inference_input_file, hparams=None):
  """Load inference data."""
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(inference_input_file, mode="rb")) as f:
    inference_data = f.read().splitlines()

  if hparams and hparams.inference_indices:
    inference_data = [inference_data[i] for i in hparams.inference_indices]

  return inference_data

def modelInit():
    
  out_dir='tmp/nmt_attention_model'
  
  print("loading parameters")
  hparams = utils.load_hparams(out_dir)
  ckpt = tf.train.latest_checkpoint(out_dir)
  
  if not hparams.attention:
    model_creator = nmt_model.Model
  elif hparams.attention_architecture == "standard":
    model_creator = attention_model.AttentionModel
  elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
    model_creator = gnmt_model.GNMTModel
  else:
    raise ValueError("Unknown model architecture")

  infer_model = model_helper.create_infer_model(model_creator, hparams, scope=None)
  return infer_model, ckpt, hparams
  
infer_model, ckpt, hparams=modelInit()
    
def inference(infer_data):
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth = True)
  with tf.Session(
      graph=infer_model.graph, config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True,log_device_placement=True)) as sess:
    loaded_infer_model = model_helper.load_model(
        infer_model.model, ckpt, sess, "infer")
    
    sess.run(
        infer_model.iterator.initializer,
        feed_dict={
            infer_model.src_placeholder: [infer_data],
            infer_model.batch_size_placeholder: hparams.infer_batch_size
        })
    
    translation=decode_inference_indices(
          loaded_infer_model,
          sess)
   
  return translation


if __name__=="__main__":
   sentence=inference("good morning".encode("utf-8")) 
   print(sentence.decode("utf-8"))
   print("*******")
   sentence=inference("good morning".encode("utf-8")) 
   print(sentence.decode("utf-8"))







