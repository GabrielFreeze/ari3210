import warnings 

warnings.filterwarnings('ignore')

import os
import time
import pylab
import random
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from multiprocessing import Process, cpu_count




N_MELS = 256
MEL_SPEC_FRAME_SIZE = 1024
SAMPLING_RATE = 16_000


def get_speaker_paths(datapath: str) -> list[str]:
    speaker_list = []
    accent_subfolders = [f.path for f in os.scandir(datapath) if f.is_dir()]

    for accent in accent_subfolders:
        for gender in ['female', 'male']:
            for speaker in os.listdir(os.path.join(accent, gender)):

                if not speaker.startswith('.'):
                    speaker_list.append((speaker,os.path.join(accent, gender, speaker)))

    return speaker_list
def get_wav_files(datapath: str) -> list[str]:
    return [file for file in os.listdir(datapath) if file.endswith('.wav')]
def plot_melspec(melspec, fs):
    plt.figure(figsize=(20, 8))
    plt.xlabel('Time')
    plt.ylabel('Mel-Frequency')
    librosa.display.specshow(melspec,
                             y_axis='mel',
                             fmax=fs / 2,
                             sr=fs,
                             hop_length=int(MEL_SPEC_FRAME_SIZE / 2),
                             x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()
def mel_spectogram_chunks(wavfile_path: str, chunk_seconds: float,
                          plot: bool = False) -> (librosa.feature.melspectrogram):

    # Load .wav file
    sig, fs = librosa.load(wavfile_path, sr=SAMPLING_RATE)

    # Normalise between [-1,1]
    sig /= np.max(np.abs(sig), axis=0)

    # Determine the number of chunk samples
    samples = fs*chunk_seconds if chunk_seconds else len(sig)
    samples_elapsed = 0
    
    melspec_chunks = []
    
    while samples_elapsed < len(sig):
        melspec = librosa.feature.melspectrogram(y=sig[samples_elapsed:(samples_elapsed + samples)],
                                                 sr=fs,
                                                 center=True,
                                                 n_fft=MEL_SPEC_FRAME_SIZE,
                                                 hop_length=int(
                                                     MEL_SPEC_FRAME_SIZE / 2),
                                                 n_mels=N_MELS)
        melspec = librosa.power_to_db(melspec, ref=np.max)
        melspec_chunks.append(melspec)
        samples_elapsed += samples
    
    return melspec_chunks
def worker(start:int=0, end:int=1):
    
    #This is so 1 implies that all of the rest of the..
    # array will be used, not up to the second item.
    if end == 1:
        speaker_path.append(speaker_path[-1])
    
    #For every speaker
    for i,(speaker, path) in enumerate(speaker_path[start:end]):

        #For every .wav file for that speaker
        for file in get_wav_files(path):

            #Get path to .wav file
            filepath = os.path.join(path,file)

            # Slice into 3 seconds chunks
            mel_chunks = mel_spectogram_chunks(wavfile_path=filepath, chunk_seconds=3, plot=True)

            # Remove last item because it is not 3 seconds.
            mel_chunks = mel_chunks[:-1]

            #Save every 3 second chunk into an image in the respect speaker folder
            for i,mel in enumerate(mel_chunks):
                # print(f'{speaker} - {file}: {str(i+1).zfill(3)}                                    ',end='\r')

                plt.clf() #Important. Without this time to save plot grows linearly.
                pylab.axis('off') #Remove Axis
                pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) #Remove white padding

                count[speaker] += 1
                save_path = os.path.join(folder,speaker,f'{str(count[speaker]).zfill(3)}.jpg')

                librosa.display.specshow(mel)

                pylab.savefig(save_path, bbox_inches=None, pad_inches=0, dpi=15) #Save Image  

# For loop that goes through the relative paths of every .wav file
corpus_path = '..\\corpus'
speaker_path = get_speaker_paths(corpus_path)

#For every speaker create a folder
folder = os.path.join('..','data')

try: os.mkdir(folder)    
except: pass

for speaker,_ in speaker_path:
    try:os.mkdir(os.path.join(folder,speaker))
    except: pass

pylab.axis('off') #Remove Axis
pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) #Remove white padding

#For naming the images
count = {speaker:0 for speaker,_ in speaker_path}


#Begin operations
if __name__ == '__main__':

    n = len(speaker_path)
    m = cpu_count() #Number of Concurrent Processes

    jobs = [Process(target=worker, args=(int(i*(n/m)), int((i+1)*(n/m))))
            for i in range(m)]
    
    s = time.time()
    for p in jobs:
        p.start()

    for p in jobs:
        p.join()

    print(f'Finished multi-processing operation in {(time.time()-s)//60}mins')
    
    ######### MULTI-PROCESSING FINISHED #########
    
    #For every split create a folder

    folder = os.path.join('..','data')

    for split in ['train','val','test']:
        try:os.mkdir(os.path.join(folder,split))
        except: pass
        
        for speaker,_ in speaker_path:
            try: os.mkdir(os.path.join(folder,split,speaker))
            except: pass

    #Perform the split by moving (75%,15%,15%) from every speaker folder
    #to the respective folder in the split

    for speaker in os.listdir(folder):
        
        if speaker in ['train','val','test']:
            continue
        
        
        #Create a shuffled list of all image names in speaker
        random.shuffle(idx := [name for name in os.listdir(os.path.join(folder,speaker))])
        

        _75 = int(len(idx)*0.75)
        _15 = int(len(idx)*0.15)
        
        #Move first 75% to train/speaker
        for file in idx[:_75]:
            os.replace(os.path.join(folder,speaker,file),
                    os.path.join(folder,'train',speaker,file))
            
        #Move second 15% to train/speaker
        for file in idx[_75:-_15]:
            os.replace(os.path.join(folder,speaker,file),
                    os.path.join(folder,'val',speaker,file))
            
        #Move last 15% to test/speaker
        for file in idx[-_15:]:
            os.replace(os.path.join(folder,speaker,file),
                    os.path.join(folder,'test',speaker,file))


        #Finally remove all the empty speaker folders
        for speaker,_ in speaker_path:
            try: os.rmdir(os.path.join(folder,speaker))
            except: pass
    

    

    


    

            

    
        


          

    
