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
from __future__ import print_function

import os
import tensorflow as tf
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from threading import Thread

import random
import codecs
import attention_model
import gnmt_model
import model as nmt_model
import model_helper
import pandas as pd
from utils import misc_utils as utils




EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz?!.\' ' 
EN_BLACKLIST = '"#$%&()*+,-/:;<=>@[\\]^_`{|}~\''
    
def start(bot, update):
   update.message.reply_text('Hi! I am Jiango Wango. How may I assist you today?')


def help(bot, update):
    update.message.reply_text('Help!')

       
def filter_sentence(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])


def decode_inference_indices(model, sess):
     nmt_outputs, infer_summary = model.decode(sess)
     assert nmt_outputs.shape[0] == 1
     
     output = nmt_outputs[0, :].tolist()

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
 
def removeDuplicateWords(oldsentence):
    tempS=oldsentence.split()
    sentence = []
    [sentence.append(x) for x in tempS if x not in sentence]
    sentence=' '.join(sentence)
    return sentence


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
  with tf.Session(
      graph=infer_model.graph, config=utils.get_config_proto()) as sess:
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

sentence=inference("good morning".encode("utf-8")) 
sentence=sentence.decode("utf-8")
sentence = sentence.split('</s>')[0]
c=[b for b in sentence.split(' ') if not "<unk>" in b]
sentence=' '.join(word for word in c)
print(sentence)
print("*******")
lastsentence = sentence 
df= pd.read_csv('wordlist.txt', header=None) 
index1=random.randint(0,len(df))
print(df.values[index1][0])
def echo(bot, update):
        global lastsentence
        global df
        text=str.lower(update.message.text)
        print(lastsentence)
        if '@w' in text:
            text=text.split('@w')[-1]
            text=filter_sentence(text, EN_WHITELIST)
            sentence=inference(text)
            sentence=sentence.decode("utf-8")
            sentence = sentence.split('</s>')[0]
            
            print(sentence)
            if sentence==lastsentence:
                print("last sentence same as previous")
                index1=random.randint(0,len(df))
                index2=random.randint(0,len(df))
                text=' '.join(word for word in [text, str.lower(df.values[index1][0]), str.lower(df.values[index2][0])])
                print("two random words: ", str.lower(df.values[index1][0]), str.lower(df.values[index2][0]))
                print(text)
                text=filter_sentence(text, EN_WHITELIST)
                sentence=inference(text)
                sentence=sentence.decode("utf-8")
                sentence = sentence.split('</s>')[0] 
                print(sentence)
                
            lastsentence = sentence 
            
            if '<unk>' in sentence:
                c=[b for b in sentence.split(' ') if not '<unk>' in b]
                
                print("after removing <unk>: ",c)
                index1=random.randint(0,len(df))
                print("adding index: ", str.lower(df.values[index1][0]))
                
                sentence=' '.join(word for word in c)
                sentence=sentence + ' ' + str.lower(df.values[index1][0])
                print(sentence)
            if ('hey' in text or 'hi' in text or 'bye' in text or 'good morning' in text or 'hello' in text or 'goodnight' in text or 'wango' in text or 'fuck' in text or 'you' in text):
                sentence = sentence+' '+str(update.message.from_user.first_name)
            bot.send_message(chat_id=update.message.chat_id, text=sentence)
            with tf.gfile.GFile(os.getcwd()+'/TelegramConversation.en', mode="ab") as vocab_file:
                vocab_file.write(str(update.message.chat_id).encode('utf-8')+b":--:"+str(update.message.from_user.id).encode('utf-8')+b":--:"+str(update.message.from_user.first_name).encode('utf-8')+b":--:"+str(update.message.message_id).encode('utf-8')+b":--:"+text.encode('utf-8') + b"\n")
                vocab_file.write(str(update.message.chat_id).encode('utf-8')+b":--:"+str(update.message.from_user.id).encode('utf-8')+b":--:"+str('Wango').encode('utf-8')+b":--:"+str(update.message.message_id).encode('utf-8')+b":--:"+sentence.encode('utf-8') + b"\n")  
        else:
    		
            with tf.gfile.GFile(os.getcwd()+'/TelegramConversation.en', mode="ab") as vocab_file:
                vocab_file.write(str(update.message.chat_id).encode('utf-8')+b":--:"+str(update.message.from_user.id).encode('utf-8')+b":--:"+str(update.message.from_user.first_name).encode('utf-8')+b":--:"+str(update.message.message_id).encode('utf-8')+b":--:"+text.encode('utf-8') + b"\n")
    			
                
def decode():
       

    updater = Updater("<InsertyourTelegramBotTokenhere>")
    
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
    dp.add_handler(CommandHandler("stopn", stopcall))
    	#dp.add_handler(CommandHandler('r', restart, filters=Filters.user(username='@Sujito')))
    	# on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))
    
    	# Start the Bot
    updater.start_polling()
#    testing= decodebotdata("checking checking", model, sess)
    
    updater.idle()

def main():
    decode()
 

if __name__ == "__main__":
  main()
