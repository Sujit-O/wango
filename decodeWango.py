# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging

import numpy as np
import tensorflow as tf

import data_utils
import seq2seq_model
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from threading import Thread



tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 24, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 12, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 1000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 1000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [ (5, 8), (8,12), (12, 14), (16,20),(20,30)]

en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.from" % FLAGS.from_vocab_size)
fr_vocab_path = os.path.join(FLAGS.data_dir,
                             "vocab%d.to" % FLAGS.to_vocab_size)
en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
_, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz?!.\' ' 
EN_BLACKLIST = '"#$%&()*+,-/:;<=>@[\\]^_`{|}~\''
    
def start(bot, update):
   update.message.reply_text('Hi! I am Jiango Wango. How may I assist you today?')


def help(bot, update):
    update.message.reply_text('Help!')

       
def filter_sentence(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])

def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.from_vocab_size,
      FLAGS.to_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model



def removeDuplicateWords(oldsentence):
    tempS=oldsentence.split()
    sentence = []
    [sentence.append(x) for x in tempS if x not in sentence]
    sentence=' '.join(sentence)
    return sentence

def decodebotdata(sentence,model, sess):
    # Get token-ids for the input sentence.
          sys.stdout.write(sentence)
          sys.stdout.flush()
          token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
          # Which bucket does it belong to?
          bucket_id = len(_buckets) - 1
          for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
              bucket_id = i
              break
          else:
            logging.warning("Sentence truncated: %s", sentence)
    
          # Get a 1-element batch to feed the sentence to the model.
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              {bucket_id: [(token_ids, [])]}, bucket_id)
          # Get output logits for the sentence.
          _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)
          # This is a greedy decoder - outputs are just argmaxes of output_logits.
          outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
          # If there is an EOS symbol in outputs, cut them at that point.
          if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
          # Print out French sentence corresponding to outputs.
          sentence=" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs])
          sentence=removeDuplicateWords(sentence)
          sys.stdout.write(":")
          sys.stdout.write(sentence)
          sys.stdout.flush()
          return sentence  

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth = True)
sess= tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,  allow_soft_placement=True, log_device_placement=True)) 
#    sess.run()
# Create model and load parameters.
model = create_model(sess, True)
model.batch_size = 1  # We decode one sentence at a time.

def echo(bot, update):
        global model
        global sess
        text=str.lower(update.message.text)
        
        if '@w' in text:
            text=text.split('@w')[-1]
#            sys.stdout.write(text)
#            sys.stdout.flush()
#            bot.send_message(chat_id=update.message.chat_id, text=text)
            text=filter_sentence(text, EN_WHITELIST)
            sentence=decodebotdata(text, model, sess)
#            sys.stdout.write(sentence)
#            sys.stdout.flush()
            bot.send_message(chat_id=update.message.chat_id, text=sentence+'.')
            with tf.gfile.GFile(os.getcwd()+'/TelegramConversation.en', mode="ab") as vocab_file:
                vocab_file.write(str(update.message.chat_id).encode('utf-8')+b":--:"+str(update.message.from_user.id).encode('utf-8')+b":--:"+str(update.message.from_user.first_name).encode('utf-8')+b":--:"+str(update.message.message_id).encode('utf-8')+b":--:"+text.encode('utf-8') + b"\n")
                vocab_file.write(str(update.message.chat_id).encode('utf-8')+b":--:"+str(update.message.from_user.id).encode('utf-8')+b":--:"+str('Wango').encode('utf-8')+b":--:"+str(update.message.message_id).encode('utf-8')+b":--:"+sentence.encode('utf-8') + b"\n")  
        else:
    		#bot.send_message(chat_id=update.message.chat_id, text='Sorry I do not know how to respond to this one yet!')
            with tf.gfile.GFile(os.getcwd()+'/TelegramConversation.en', mode="ab") as vocab_file:
                vocab_file.write(str(update.message.chat_id).encode('utf-8')+b":--:"+str(update.message.from_user.id).encode('utf-8')+b":--:"+str(update.message.from_user.first_name).encode('utf-8')+b":--:"+str(update.message.message_id).encode('utf-8')+b":--:"+text.encode('utf-8') + b"\n")
    			#vocab_file.write(str(update.user.first_name).encode('utf-8')+b":"+text.encode('utf-8') + b"\n")
                
def decode():
    global model
    global sess

    

    updater = Updater("<Insertyourtelegrambottokenhere>")
    
    	# Get the dispatcher to register handlers
    dp = updater.dispatcher
    	
    def stop():
        """Gracefully stop the Updater and exit"""
        updater.stop()
#        sess.close()
        os._exit(1)
    
    def stopcall(bot, update):
        update.message.reply_text('Bot is Stopping...')
        Thread(target=stop).start()
 
      	  
    	# on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("stopp", stopcall))
    	#dp.add_handler(CommandHandler('r', restart, filters=Filters.user(username='@Sujito')))
    	# on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))
    
    	# Start the Bot
    updater.start_polling()
#    testing= decodebotdata("checking checking", model, sess)
    
    updater.idle()

def main(_):
    decode()
 

if __name__ == "__main__":
  tf.app.run()
