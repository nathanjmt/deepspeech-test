from deepspeech import Model, version
from jiwer import wer
import librosa as lr
import numpy as np
import pandas as pd
import os 

#set the scorers and models for the respective languages
en_scorer = "scorers\deepspeech-0.9.3-models.scorer"
en_model = "models\deepspeech-0.9.3-models.pbmm"

es_scorer = "scorers\kenlm_es.scorer"
es_model = "models\output_graph_es.pbmm"

it_scorer = "scorers\kenlm_it.scorer"
it_model = "models\output_graph_it.pbmm"

#load the audio files for testing into a list
audio_files = ["EN\checkin.wav","EN\parents.wav","EN\suitcase.wav","EN\what_time.wav","EN\where.wav",
                "ES\checkin_es.wav","ES\parents_es.wav","ES\suitcase_es.wav","ES\what_time_es.wav","ES\where_es.wav",
                "IT\checkin_it.wav","IT\parents_it.wav","IT\suitcase_it.wav","IT\what_time_it.wav","IT\where_it.wav",
                "EN\my_sentence1.wav", "EN\my_sentence2.wav"]


assert os.path.exists(en_scorer), en_scorer + "not found. Perhaps you need to download a scroere  from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"
assert os.path.exists(en_model), en_model + "not found. Perhaps you need to download a  model from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"
for audio in audio_files:   
    assert os.path.exists(audio), audio + "does not exist"

en_ds = Model(en_model)
en_ds.enableExternalScorer(en_scorer)

es_ds = Model(es_model)
es_ds.enableExternalScorer(es_scorer)

it_ds = Model(it_model)
it_ds.enableExternalScorer(it_scorer)

#transcripts for testing results
transcripts = {
   "EN\checkin.wav" : "where is the check in desk",
   "EN\parents.wav" : "i have lost my parents",
   "EN\suitcase.wav" : "please i have lost my suitcase",
   "EN\what_time.wav" : "what time is my plane",
   "EN\where.wav" : "where are the restaurants and shops",
   "ES\checkin_es.wav" : "Donde estan los mostradores",
   "ES\parents_es.wav" : "he perdido a mis padres",
   "ES\suitcase_es.wav" : "por favor he perdido mi maleta",
   "ES\what_time_es.wav" : "a que hora es mi avion",
   "ES\where_es.wav" : "donde estan los restaurantes y las tiendas",
   "IT\checkin_it.wav" : "dove e il bacone",
   "IT\parents_it.wav" : "ho perso i miei genitori",
   "IT\suitcase_it.wav" : "per favore ho perso la mia valigia",
   "IT\what_time_it.wav" : "a che ora e il mio aereo",
   "IT\where_it.wav" : "dove sono i ristoranti e i negozi",
   "EN\my_sentence1.wav" : "how do i get to my boarding gate", 
   "EN\my_sentence2.wav" : "i need to print my boarding pass"
}

#initialise a list to load the results of the test
results = []

for audio_file in audio_files:
    if audio_file[:2] == 'ES':
        desired_sample_rate = es_ds.sampleRate()
        audio = lr.load(audio_file, sr=desired_sample_rate)[0]
        processed_audio = audio*2 #increase amplitude by 2
        processed_audio = (processed_audio * 32767).astype(np.int16) # scale from -1 to 1 to +/-32767
        res = es_ds.stt(processed_audio)
        error = wer(transcripts[audio_file], res)*100 #get WER as percentage
        #add results to list
        results.append('Spanish')
        results.append(audio_file[3:])
        results.append(str(round(error,1)) + "%")

    elif audio_file[:2] == 'IT':
        desired_sample_rate = it_ds.sampleRate()
        audio = lr.load(audio_file, sr=desired_sample_rate)[0]
        processed_audio = audio*2 #increase amplitude by 2
        processed_audio = (processed_audio * 32767).astype(np.int16) # scale from -1 to 1 to +/-32767
        res = it_ds.stt(processed_audio)
        error = wer(transcripts[audio_file], res)*100 #get WER as percentage
        #add results to list
        results.append('Italian')    
        results.append(audio_file[3:])
        results.append(str(round(error,1)) + "%")

    else:
        desired_sample_rate = en_ds.sampleRate()
        audio = lr.load(audio_file, sr=desired_sample_rate)[0]
        processed_audio = audio*1.75 #increase amplitude by 2
        processed_audio = (processed_audio * 32767).astype(np.int16) # scale from -1 to 1 to +/-32767
        res = en_ds.stt(processed_audio)
        error = wer(transcripts[audio_file], res)*100 #get WER as percentage
        #add results to list
        results.append('English')
        results.append(audio_file[3:])
        results.append(str(round(error,1)) + "%")


#convert list to np array, reshape and generate table
results_np = np.array(results)
table = results_np.reshape(17,3)
print(pd.DataFrame(table, columns = ['Language', "File", "WER"]))



